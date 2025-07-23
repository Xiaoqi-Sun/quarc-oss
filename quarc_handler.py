import os
import json
import torch
import logging
from typing import Any, Dict, List
from pathlib import Path

# Import QUARC components
from quarc.predictors.model_factory import load_models_from_yaml
from quarc.predictors.multistage_predictor import EnumeratedPredictor
from quarc.models.modules.agent_encoder import AgentEncoder
from quarc.models.modules.agent_standardizer import AgentStandardizer
from quarc.models.modules.rxn_encoder import ReactionClassEncoder
from quarc.data.datapoints import ReactionDatum
from chemprop.featurizers import CondensedGraphOfReactionFeaturizer

logger = logging.getLogger(__name__)

class QuarcHandler:
    """QUARC Handler for TorchServe"""

    def __init__(self):
        self._context = None
        self.initialized = False

        # Model components
        self.predictor = None
        self.device = None
        self.model_type = None  # 'ffn' or 'gnn'

        # Supporting components
        self.agent_encoder = None
        self.agent_standardizer = None
        self.rxn_encoder = None
        self.featurizer = None

    def initialize(self, context):
        """Initialize QUARC models and supporting components"""
        self._context = context
        self.manifest = context.manifest

        properties = context.system_properties
        model_dir = properties.get("model_dir")
        logger.info(f"Model directory: {model_dir}")
        logger.info(f"Contents: {os.listdir(model_dir)}")

        # Set device
        if torch.cuda.is_available():
            self.device = torch.device("cuda:" + str(properties.get("gpu_id", 0)))
        else:
            self.device = torch.device("cpu")
        logger.info(f"Using device: {self.device}")

        try:
            # Load model configuration
            config_files = [f for f in os.listdir(model_dir) if f.endswith('_pipeline.yaml')]
            if not config_files:
                raise FileNotFoundError("No pipeline config file found (ffn_pipeline.yaml or gnn_pipeline.yaml)")

            config_file = os.path.join(model_dir, config_files[0])
            self.model_type = 'ffn' if 'ffn' in config_files[0] else 'gnn'
            logger.info(f"Using {self.model_type.upper()} models with config: {config_file}")

            # Load models using QUARC's model factory
            models, model_types, weights = load_models_from_yaml(
                config_path=config_file,
                device=self.device
            )
            logger.info(f"Loaded {len(models)} models: {list(models.keys())}")

            # Initialize supporting components
            self._initialize_encoders(model_dir)

            # Create the enumerated predictor
            self.predictor = EnumeratedPredictor(
                models=models,
                model_types=model_types,
                weights=weights,
                agent_encoder=self.agent_encoder,
                agent_standardizer=self.agent_standardizer,
                rxn_encoder=self.rxn_encoder,
                featurizer=self.featurizer if self.model_type == 'gnn' else None,
                device=self.device
            )

            self.initialized = True
            logger.info("QUARC handler initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize QUARC handler: {str(e)}")
            raise e

    def _initialize_encoders(self, model_dir):
        """Initialize agent encoder, standardizer, and reaction class encoder"""
        try:
            # Agent encoder
            agent_encoder_path = os.path.join(model_dir, "agent_encoder_list.json")
            if os.path.exists(agent_encoder_path):
                self.agent_encoder = AgentEncoder(class_path=agent_encoder_path)
                logger.info("Loaded agent encoder")
            else:
                logger.warning(f"Agent encoder file not found: {agent_encoder_path}")

            # Agent standardizer
            rules_path = os.path.join(model_dir, "agent_rules_v1.json")
            other_dict_path = os.path.join(model_dir, "agent_other_dict.json")
            if os.path.exists(rules_path) and os.path.exists(other_dict_path):
                self.agent_standardizer = AgentStandardizer(
                    conv_rules=rules_path,
                    other_dict=other_dict_path
                )
                logger.info("Loaded agent standardizer")
            else:
                logger.warning("Agent standardizer files not found")

            # Reaction class encoder (optional - may not be needed for serving)
            namerxn_path = os.path.join(model_dir, "namerxn_classes.json")
            if os.path.exists(namerxn_path):
                self.rxn_encoder = ReactionClassEncoder(class_path=namerxn_path)
                logger.info("Loaded reaction class encoder")
            else:
                logger.info("No reaction class encoder file found - will use provided rxn_class directly")

            # Molecular featurizer for GNN models
            if self.model_type == 'gnn':
                self.featurizer = CondensedGraphOfReactionFeaturizer(mode_="REAC_DIFF")
                logger.info("Initialized molecular featurizer for GNN models")

        except Exception as e:
            logger.error(f"Failed to initialize encoders: {str(e)}")
            # Don't raise here - allow partial initialization

    def preprocess(self, data: List) -> List[ReactionDatum]:
        """Preprocess input data to ReactionDatum objects"""
        try:
            # Extract JSON data from request
            input_data = data[0].get("body") or data[0].get("data")
            if isinstance(input_data, (bytes, bytearray)):
                input_data = input_data.decode('utf-8')

            if isinstance(input_data, str):
                input_data = json.loads(input_data)

            # Handle both single reaction and batch of reactions
            reactions = input_data.get("reactions", [input_data])
            if not isinstance(reactions, list):
                reactions = [reactions]

            # Convert to ReactionDatum objects
            reaction_data = []
            for i, rxn in enumerate(reactions):
                # Validate required fields
                if "rxn_smiles" not in rxn:
                    raise ValueError(f"Missing 'rxn_smiles' in reaction {i}")
                if "rxn_class" not in rxn:
                    logger.warning(f"Missing 'rxn_class' in reaction {i}, using default")
                    rxn["rxn_class"] = "1.0.0"  # Default class

                # Create ReactionDatum
                datum = ReactionDatum(
                    document_id=rxn.get("doc_id", f"reaction_{i}"),
                    rxn_class=rxn["rxn_class"],
                    date=rxn.get("date", ""),
                    rxn_smiles=rxn["rxn_smiles"],
                    reactants=[],  # Will be parsed from rxn_smiles if needed
                    products=[],   # Will be parsed from rxn_smiles if needed
                    agents=[],     # Will be predicted
                    temperature=None  # Will be predicted
                )
                reaction_data.append(datum)

            logger.info(f"Preprocessed {len(reaction_data)} reactions")
            return reaction_data

        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            raise e

    def inference(self, reaction_data: List[ReactionDatum]) -> List[Dict]:
        """Run QUARC inference on reaction data"""
        try:
            results = []

            for datum in reaction_data:
                try:
                    # Run prediction using EnumeratedPredictor
                    prediction_list = self.predictor.predict(datum, top_k=10)

                    # Convert PredictionList to dictionary format
                    result = {
                        "doc_id": prediction_list.document_id,
                        "rxn_class": prediction_list.rxn_class,
                        "rxn_smiles": prediction_list.rxn_smiles,
                        "predictions": []
                    }

                    # Format predictions
                    for pred in prediction_list.predictions:
                        formatted_pred = {
                            "rank": pred.rank,
                            "score": float(pred.score),
                            "agents": pred.agents,
                            "temperature": pred.temperature,
                            "reactant_amounts": pred.reactant_amounts,
                            "agent_amounts": [
                                {
                                    "agent": amt.agent,
                                    "amount_range": amt.amount_range
                                } for amt in pred.agent_amounts
                            ]
                        }
                        result["predictions"].append(formatted_pred)

                    results.append(result)
                    logger.info(f"Generated {len(prediction_list.predictions)} predictions for {datum.document_id}")

                except Exception as e:
                    logger.error(f"Inference failed for reaction {datum.document_id}: {str(e)}")
                    # Return empty result for failed reactions
                    results.append({
                        "doc_id": datum.document_id,
                        "rxn_class": datum.rxn_class,
                        "rxn_smiles": datum.rxn_smiles,
                        "predictions": [],
                        "error": str(e)
                    })

            return results

        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            raise e

    def postprocess(self, data: List[Dict]) -> List[List[Dict]]:
        """Postprocess inference results"""
        return [data]

    def handle(self, data, context) -> List[List[Dict[str, Any]]]:
        """Main handler method that orchestrates the prediction pipeline"""
        self._context = context

        if not self.initialized:
            raise RuntimeError("Handler not initialized")

        try:
            # Preprocess input data
            reaction_data = self.preprocess(data)

            # Run inference
            predictions = self.inference(reaction_data)

            # Postprocess results
            output = self.postprocess(predictions)

            logger.info(f"Successfully processed {len(reaction_data)} reactions")
            return output

        except Exception as e:
            logger.error(f"Handler execution failed: {str(e)}")
            # Return error response
            error_response = [{
                "error": str(e),
                "message": "QUARC prediction failed"
            }]
            return [error_response]

# Required for TorchServe
_service = QuarcHandler()

def handle(data, context):
    return _service.handle(data, context)