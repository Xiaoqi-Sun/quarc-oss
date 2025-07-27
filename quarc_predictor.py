import json
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import argparse

from chemprop.featurizers import CondensedGraphOfReactionFeaturizer
from loguru import logger

import quarc_parser
from quarc.models.modules.agent_encoder import AgentEncoder
from quarc.models.modules.agent_standardizer import AgentStandardizer
from quarc.models.modules.rxn_encoder import ReactionClassEncoder
from quarc.data.datapoints import AgentRecord, ReactionDatum
from quarc.data.eval_datasets import EvaluationDatasetFactory
from quarc.data.binning import BinningConfig
from quarc.utils.smiles_utils import parse_rxn_smiles
from quarc.predictors.model_factory import load_models_from_yaml
from quarc.predictors.multistage_predictor import EnumeratedPredictor


def prepare_reaction_data(input_data: list[dict[str, Any]]) -> list[ReactionDatum]:
    """Convert input data to ReactionDatum objects"""

    reactions = []
    for i, item in enumerate(input_data):  # FIXME: check input data format
        rxn_smiles = item["rxn_smiles"]
        reactants, agents, products = parse_rxn_smiles(rxn_smiles)

        reactions.append(
            ReactionDatum(
                rxn_smiles=rxn_smiles,
                reactants=[AgentRecord(smiles=r, amount=None) for r in reactants],
                agents=[AgentRecord(smiles=a, amount=None) for a in agents],
                products=[AgentRecord(smiles=p, amount=None) for p in products],
                rxn_class="unknown",
                document_id=f"reaction_{i}",
                date=None,
                temperature=None,
            )
        )

    return reactions


class QuarcPredictor:
    """QUARC predictor wrapper. Used exisiting src/quarc/predictors logic."""

    def __init__(self, args):
        self.args = args
        self.config_path = args.config_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # possible ones
        # self.model_name = args.model_name
        # self.data_name = args.data_name
        # self.log_file = args.log_file
        # self.processed_data_path = args.processed_data_path
        # self.model_path = args.model_path
        # self.test_output_path = args.test_output_path
        # self.test_file = args.test_file

        self._initialize_components()
        self._load_models()

        logger.info("QUARC predictor initialized successfully")

    def _initialize_components(self):
        self.agent_encoder = AgentEncoder(
            class_path=self.args.processed_data_dir / "agent_encoder/agent_encoder_list.json"
        )
        self.agent_standardizer = AgentStandardizer(
            conv_rules=self.args.processed_data_dir / "agent_encoder/agent_rules_v1.json",
            other_dict=self.args.processed_data_dir / "agent_encoder/agent_other_dict.json",
        )
        self.featurizer = CondensedGraphOfReactionFeaturizer(mode_="REAC_DIFF")
        self.binning_config = BinningConfig.default()  # TODO: allow to be set by user

    def _load_models(self):
        config_file = self.config_path
        if not config_file.exists():
            raise FileNotFoundError(f"Model config not found: {config_file}")

        models, model_types, weights = load_models_from_yaml(config_file, self.device)

        self.predictor = EnumeratedPredictor(
            agent_model=models["agent"],
            temperature_model=models["temperature"],
            reactant_amount_model=models["reactant_amount"],
            agent_amount_model=models["agent_amount"],
            model_types=model_types,
            agent_encoder=self.agent_encoder,
            device=self.device,
            weights=weights["use_top_5"],
            use_geometric=weights["use_geometric"],
        )

    def _format_prediction_results(self, predictions, top_k: int = 5) -> Dict[str, Any]:
        """Format prediction results into structured output"""

        temp_labels = self.binning_config.get_bin_labels("temperature")
        reactant_labels = self.binning_config.get_bin_labels("reactant")
        agent_labels = self.binning_config.get_bin_labels("agent")

        results = {
            "doc_id": predictions.doc_id,
            "rxn_class": predictions.rxn_class,
            "rxn_smiles": predictions.rxn_smiles,
            "predictions": [],
        }

        reactants_smiles, _, _ = parse_rxn_smiles(predictions.rxn_smiles)

        for i, pred in enumerate(predictions.predictions[:top_k]):
            agent_smiles = self.agent_encoder.decode(pred.agents)
            temp_label = temp_labels[pred.temp_bin]
            reactant_labels_list = [reactant_labels[bin_idx] for bin_idx in pred.reactant_bins]

            agent_amounts = []
            for agent_idx, bin_idx in pred.agent_amount_bins:
                agent_smi = self.agent_encoder.decode([agent_idx])[0]
                amount_label = agent_labels[bin_idx]
                agent_amounts.append({"agent": agent_smi, "amount_range": amount_label})

            reactant_amounts = []
            for reactant_smi, reactant_label in zip(reactants_smiles, reactant_labels_list):
                reactant_amounts.append({"reactant": reactant_smi, "amount_range": reactant_label})

            prediction = {
                "rank": i + 1,
                "score": pred.score,
                "agents": agent_smiles,
                "temperature": temp_label,
                "reactant_amounts": reactant_amounts,
                "agent_amounts": agent_amounts,
                "raw_scores": pred.meta if hasattr(pred, "meta") else {},
            }
            results["predictions"].append(prediction)

        return results

    def predict_batch(
        self, input_data: list[dict[str, Any]], top_k: int = 5
    ) -> list[dict[str, Any]]:

        reactions = prepare_reaction_data(input_data)
        dataset = EvaluationDatasetFactory.for_inference(
            data=reactions,
            agent_standardizer=self.agent_standardizer,
            agent_encoder=self.agent_encoder,
            featurizer=self.featurizer,
        )

        # Run predictions
        all_results = []
        for reaction in dataset:
            predictions = self.predictor.predict(reaction, top_k=top_k)
            result = self._format_prediction_results(predictions, top_k=top_k)
            all_results.append(result)

        return all_results

    def predict(self, smiles_list: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Simple prediction method that takes a list of SMILES strings
        
        Args:
            smiles_list: List of reaction SMILES strings
            top_k: Number of top predictions to return
            
        Returns:
            List of prediction results, one per SMILES
        """
        # Convert SMILES list to the format expected by predict_batch
        input_data = []
        for i, rxn_smiles in enumerate(smiles_list):
            input_data.append({
                "rxn_smiles": rxn_smiles,
                "rxn_class": "0.0.0",  # Default for open-source
                "doc_id": f"reaction_{i}"
            })
        
        return self.predict_batch(input_data, top_k=top_k)

    def predict_from_file(self, input_file: str, output_file: str, top_k: int = 5):
        """file-based prediction"""

        with open(input_file, "r") as f:
            input_data = json.load(f)

        results = self.predict_batch(input_data, top_k=top_k)

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Predictions saved to: {output_file}")


# def create_predictor_for_serving(
#     config_path: str = "ffn_pipeline.yaml", device: str = "auto", use_reaction_class: bool = True
# ) -> QuarcPredictor:
#     """
#     Create QUARC predictor instance for serving applications

#     Args:
#         config_path: Pipeline config file
#         device: Device to use
#         use_reaction_class: Whether to use reaction classification

#     Returns:
#         Initialized QuarcPredictor instance
#     """

#     return QuarcPredictor(
#         config_path=config_path, device=device, use_reaction_class=use_reaction_class
#     )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("quarc inference")
    quarc_parser.add_predict_opts(parser)
    args, unknown = parser.parse_known_args()

    predictor = QuarcPredictor(args)
    predictor.predict_from_file(args.input, args.output, args.top_k)
