import json
import torch
from pathlib import Path
from typing import Any
import argparse

from chemprop.featurizers import CondensedGraphOfReactionFeaturizer
from loguru import logger

import quarc_parser
from quarc.models.modules.agent_encoder import AgentEncoder
from quarc.models.modules.agent_standardizer import AgentStandardizer
from quarc.data.datapoints import AgentRecord, ReactionDatum
from quarc.data.eval_datasets import EvaluationDatasetFactory
from quarc.data.binning import BinningConfig
from quarc.utils.smiles_utils import parse_rxn_smiles
from quarc.predictors.base import PredictionList
from quarc.predictors.model_factory import load_models_from_yaml
from quarc.predictors.multistage_predictor import EnumeratedPredictor


def prepare_reaction_data(rxn_smiles: str) -> ReactionDatum:
    """Convert input data to ReactionDatum objects"""

    reactants, agents, products = parse_rxn_smiles(rxn_smiles)

    return ReactionDatum(
        rxn_smiles=rxn_smiles,
        reactants=[AgentRecord(smiles=r, amount=None) for r in reactants],
        agents=[AgentRecord(smiles=a, amount=None) for a in agents],
        products=[AgentRecord(smiles=p, amount=None) for p in products],
        rxn_class=None,
        document_id=None,
        date=None,
        temperature=None,
    )


class QuarcPredictor:
    """QUARC predictor wrapper. Used exisiting src/quarc/predictors logic."""

    def __init__(self, args):
        self.args = args
        self.config_path = args.config_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._initialize_components()
        self._load_models()

        logger.info("QUARC predictor initialized successfully")

    def _initialize_components(self):
        self.agent_encoder = AgentEncoder(
            class_path=Path(self.args.processed_data_dir) / "agent_encoder/agent_encoder_list.json"
        )
        self.agent_standardizer = AgentStandardizer(
            conv_rules=Path(self.args.processed_data_dir) / "agent_encoder/agent_rules_v1.json",
            other_dict=Path(self.args.processed_data_dir) / "agent_encoder/agent_other_dict.json",
        )
        self.featurizer = CondensedGraphOfReactionFeaturizer(mode_="REAC_DIFF")
        self.binning_config = BinningConfig.default()  # TODO: allow to be set by user

    def _load_models(self):
        config_file = Path(self.config_path)
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

    def _format_prediction_results(
        self, predictions: PredictionList, top_k: int = 10
    ) -> dict[str, Any]:
        """Format prediction results into structured output"""

        temp_labels = self.binning_config.get_bin_labels("temperature")
        reactant_labels = self.binning_config.get_bin_labels("reactant")
        agent_labels = self.binning_config.get_bin_labels("agent")

        results = {
            # "doc_id": predictions.doc_id,
            # "rxn_class": predictions.rxn_class,
            # "rxn_smiles": predictions.rxn_smiles,
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
                "agents": agent_smiles,
                "temperature": temp_label,
                "reactant_amounts": reactant_amounts,
                "agent_amounts": agent_amounts,
                "score": pred.score,
                # "raw_scores": pred.meta if hasattr(pred, "meta") else {},
            }
            results["predictions"].append(prediction)

        return results

    def predict_batch(self, input_smiles: list[str], top_k: int = 10) -> list[dict[str, Any]]:
        # prepare smiles to ReactionInput
        reactions = [prepare_reaction_data(s) for s in input_smiles]
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser("quarc inference")
    quarc_parser.add_predict_opts(parser)
    quarc_parser.add_data_opts(parser)
    args, unknown = parser.parse_known_args()

    predictor = QuarcPredictor(args)

    example_input = [
        "[CH3:1][O:2][C:3](=[O:4])[c:5]1[cH:6][cH:7][c:8]2[nH:9][c:10]([c:11]([c:12]2[cH:13]1)[CH2:14][c:15]1[cH:16][cH:17][c:18]([cH:19][c:20]1[Cl:21])I)[CH3:22].[CH3:23][C:24]([CH3:25])([CH3:26])[SH:27]>c1ccc(cc1)[P](c1ccccc1)(c1ccccc1)[Pd]([P](c1ccccc1)(c1ccccc1)c1ccccc1)([P](c1ccccc1)(c1ccccc1)c1ccccc1)[P](c1ccccc1)(c1ccccc1)c1ccccc1.CCCCN(CCCC)CCCC.CN(C)C=O>[CH3:1][O:2][C:3](=[O:4])[c:5]1[cH:6][cH:7][c:8]2[nH:9][c:10]([c:11]([c:12]2[cH:13]1)[CH2:14][c:15]1[cH:16][cH:17][c:18]([cH:19][c:20]1[Cl:21])[S:27][C:24]([CH3:23])([CH3:25])[CH3:26])[CH3:22]",
        "Cl.[O:1]=[C:2]1[CH2:3][CH2:4][CH:5]([C:6]([NH:7]1)=[O:8])[N:9]1[CH2:10][c:11]2[c:12]([cH:13][cH:14][cH:15][c:16]2[O:17][CH2:18][c:19]2[cH:20][cH:21][cH:22][c:23]([cH:24]2)[CH2:25]Br)[C:26]1=[O:27].[F:28][c:29]1[cH:30][cH:31][c:32]([cH:33][cH:34]1)[CH:35]1[CH2:36][CH2:37][NH:38][CH2:39][CH2:40]1>CC(C)N(CC)C(C)C.CC#N>[O:1]=[C:2]1[CH2:3][CH2:4][CH:5]([C:6]([NH:7]1)=[O:8])[N:9]1[CH2:10][c:11]2[c:16]([cH:15][cH:14][cH:13][c:12]2[C:26]1=[O:27])[O:17][CH2:18][c:19]1[cH:20][cH:21][cH:22][c:23]([cH:24]1)[CH2:25][N:38]1[CH2:37][CH2:36][CH:35]([CH2:40][CH2:39]1)[c:32]1[cH:31][cH:30][c:29]([cH:34][cH:33]1)[F:28]",
    ]
    res = predictor.predict_batch(example_input, top_k=3)
    print(res)
