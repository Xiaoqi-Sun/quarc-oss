import warnings

warnings.filterwarnings("ignore")

import os
import pickle
import time
import torch
import multiprocessing as mp
import argparse
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from pprint import pformat
import optuna
import json
import random
import sys

from src.quarc.models.modules.agent_encoder import AgentEncoder
from src.quarc.models.modules.agent_standardizer import AgentStandardizer
from chemprop.featurizers import CondensedGraphOfReactionFeaturizer
from src.quarc.data.eval_datasets import (
    EvaluationDatasetFactory,
    ReactionInput,
    UnifiedEvaluationDataset,
)
from src.quarc.predictors.model_factory import load_models_from_yaml
from src.quarc.predictors.multistage_predictor import HierarchicalPrediction
from src.quarc.predictors.precomputed_hierarchical_predictor import (
    PrecomputedHierarchicalPredictor,
)
from src.quarc.predictors.base import PredictionList
from eval_utils import check_overall_prediction


def process_chunk_hierarchical(
    chunk_data, gpu_id, pipeline_config_path, prediction_config, processed_data_dir
):
    """
    Process a chunk of data to generate hierarchical predictions.
    """
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    pipeline_name = Path(pipeline_config_path).stem
    logger.info(f"Processing chunk of {len(chunk_data)} reactions on {device} for {pipeline_name}")

    models, model_types, _ = load_models_from_yaml(pipeline_config_path, device)

    processed_data_path = Path(processed_data_dir)
    a_enc = AgentEncoder(class_path=processed_data_path / "agent_encoder/agent_encoder_list.json")
    a_standardizer = AgentStandardizer(
        conv_rules=processed_data_path / "agent_encoder/agent_rules_v1.json",
        other_dict=processed_data_path / "agent_encoder/agent_other_dict.json",
    )
    featurizer = CondensedGraphOfReactionFeaturizer(mode_="REAC_DIFF")

    chunk_dataset = EvaluationDatasetFactory.for_models(
        data=chunk_data,
        agent_standardizer=a_standardizer,
        agent_encoder=a_enc,
        featurizer=featurizer,
    )

    chunk_results = []

    for reaction in chunk_dataset:
        try:
            hier_pred = HierarchicalPrediction.from_models(
                reaction_data=reaction,
                agent_model=models["agent"],
                temperature_model=models["temperature"],
                reactant_amount_model=models["reactant_amount"],
                agent_amount_model=models["agent_amount"],
                model_types=model_types,
                agent_encoder=a_enc,
                top_k_agents=prediction_config["top_k_agents"],
                top_k_temp=prediction_config["top_k_temp"],
                top_k_reactant_amount=prediction_config["top_k_reactant_amount"],
                top_k_agent_amount=prediction_config["top_k_agent_amount"],
                device=device,
            )
            chunk_results.append(hier_pred)
        except Exception as e:
            logger.error(
                f"Error processing reaction {reaction.metadata.get('doc_id', 'unknown')}: {e}"
            )
            continue

    del models
    torch.cuda.empty_cache()

    return chunk_results


def run_hierarchical_precomputation(
    pipeline_config_path: Path | str,
    data_path: Path | str,
    output_dir: Path | str,
    processed_data_dir: Path | str,
):
    """
    Main function to precompute hierarchical predictions for one OSS pipeline.
    """
    pipeline_name = Path(pipeline_config_path).stem
    os.makedirs(output_dir, exist_ok=True)

    if "val" in str(data_path):
        output_path = Path(output_dir) / f"{pipeline_name}_hierarchical_validation.pickle"
    else:
        output_path = Path(output_dir) / f"{pipeline_name}_hierarchical.pickle"

    num_gpus = torch.cuda.device_count()
    chunk_size = 1000

    prediction_config = {
        "top_k_agents": 10,
        "top_k_temp": 2,
        "top_k_reactant_amount": 2,
        "top_k_agent_amount": 2,
    }

    logger.info(f"Hierarchical precomputation configuration for {pipeline_name}:")
    logger.info(f"Pipeline: {pipeline_name}")
    logger.info(f"Config: {pformat(prediction_config)}")
    logger.info(f"Output path: {output_path}")

    logger.info(f"Loading data from {data_path}")
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    chunks = []
    for i in range(0, len(data), chunk_size):
        chunks.append(data[i : i + chunk_size])
    logger.info(f"Split data into {len(chunks)} chunks of size ~{chunk_size}")

    all_hierarchical_results = []

    if num_gpus > 1:
        mp.set_start_method("spawn", force=True)
        pool = mp.Pool(num_gpus)

        # Submit jobs
        jobs = []
        for i, chunk in enumerate(chunks):
            gpu_id = i % num_gpus
            jobs.append(
                pool.apply_async(
                    process_chunk_hierarchical,
                    args=(
                        chunk,
                        gpu_id,
                        pipeline_config_path,
                        prediction_config,
                        processed_data_dir,
                    ),
                )
            )

        # Collect results
        for job in tqdm(jobs, desc="Processing chunks"):
            chunk_results = job.get()
            all_hierarchical_results.extend(chunk_results)

        pool.close()
        pool.join()
    else:
        # Single GPU processing
        for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
            chunk_results = process_chunk_hierarchical(
                chunk, 0, pipeline_config_path, prediction_config, processed_data_dir
            )
            all_hierarchical_results.extend(chunk_results)

    logger.info(f"Generated {len(all_hierarchical_results)} hierarchical predictions")

    # Save results
    with open(output_path, "wb") as f:
        pickle.dump(all_hierarchical_results, f)

    logger.info(f"Hierarchical predictions saved to: {output_path}")
    return output_path


def calculate_topk_accuracy_simple(
    prediction_lists: list[PredictionList],
    reaction_inputs: list[ReactionInput] | UnifiedEvaluationDataset,
    agent_encoder,
    max_k: int = 10,
):

    hit_counters = {
        "overall": {k: 0 for k in range(1, max_k + 1)},
        "overall_relaxed": {k: 0 for k in range(1, max_k + 1)},
    }

    total_reactions = len(prediction_lists)

    for pred_list, reaction_input in zip(prediction_lists, reaction_inputs):
        match_found = {criterion: False for criterion in hit_counters.keys()}

        for i, stage_pred in enumerate(pred_list.predictions[:max_k]):
            k = i + 1  # 1-indexed k

            # Check correctness
            targets = reaction_input.targets
            correctness = check_overall_prediction(stage_pred, targets, agent_encoder)
            criteria_checks = {
                "overall": correctness.is_fully_correct,
                "overall_relaxed": correctness.is_fully_correct_relaxed,
            }

            for criterion, is_correct in criteria_checks.items():
                if is_correct and not match_found[criterion]:
                    match_found[criterion] = True
                    # Update ALL k values from current position onward
                    for hit_k in range(k, max_k + 1):
                        hit_counters[criterion][hit_k] += 1

    accuracies = {
        criterion: {k: hits / total_reactions for k, hits in counter.items()}
        for criterion, counter in hit_counters.items()
    }

    return accuracies


def load_validation_data(data_path: Path, sample_size: int = 10000) -> list[ReactionInput]:
    """Load and sample validation data."""
    with open(data_path, "rb") as f:
        val_data = pickle.load(f)

    if sample_size and sample_size < len(val_data):
        random.seed(42)
        val_indices = random.sample(range(len(val_data)), sample_size)
        val_data = [val_data[i] for i in val_indices]
        logger.info(f"Sampled {len(val_data)} reactions for optimization with seed 42")

    return val_data


def optimize_pipeline_weights(
    pipeline_name: str,
    hierarchical_cache_path: Path,
    validation_data: list[ReactionInput],
    processed_data_dir: str,
    output_dir: Path | str,
    n_trials: int = 50,
    use_topk: int = 10,
    use_geometric: bool = True,
):
    """
    Optimize weights for a pipeline using precomputed hierarchical predictions.
    """
    logger.info(f"Starting weight optimization for {pipeline_name}")
    logger.info(f"Using top {use_topk} with geometric = {use_geometric}")

    # Setup
    processed_data_path = Path(processed_data_dir)
    a_enc = AgentEncoder(class_path=processed_data_path / "agent_encoder/agent_encoder_list.json")
    a_standardizer = AgentStandardizer(
        conv_rules=processed_data_path / "agent_encoder/agent_rules_v1.json",
        other_dict=processed_data_path / "agent_encoder/agent_other_dict.json",
    )

    validation_dataset = EvaluationDatasetFactory.for_baseline_with_targets(
        data=validation_data,
        agent_standardizer=a_standardizer,
        agent_encoder=a_enc,
        featurizer=None,
    )
    predictor = PrecomputedHierarchicalPredictor(
        hierarchical_cache_path=hierarchical_cache_path,
        weights={},
        use_geometric=use_geometric,
    )

    def objective(trial):
        # Update weights instead of creating new predictor
        weights = {
            "agent": trial.suggest_float("agent", 0.1, 0.5, step=0.05),
            "temperature": trial.suggest_float("temperature", 0.1, 0.5, step=0.05),
            "reactant_amount": trial.suggest_float("reactant_amount", 0.1, 0.5, step=0.05),
            "agent_amount": trial.suggest_float("agent_amount", 0.1, 0.5, step=0.05),
        }

        predictor.update_weights(weights)

        # Evaluate
        accs = calculate_topk_accuracy_simple(
            predictor.predict_many(validation_dataset, top_k=10),
            validation_dataset,
            a_enc,
            max_k=10,
        )
        acc_to_use = accs["overall"][use_topk]
        logger.debug(f"Trial {trial.number} overall accs: {accs['overall']}")
        return acc_to_use

    study = optuna.create_study(study_name=f"optimize_{pipeline_name}", direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    logger.info(f"{pipeline_name}: {study.best_params} -> {study.best_value:.4f}")

    # Save results
    results_path = Path(output_dir) / f"{pipeline_name}_weights_usek{use_topk}.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "pipeline": pipeline_name,
                "best_params": study.best_params,
                "best_value": study.best_value,
                "cache_path": str(hierarchical_cache_path),
                "use_geometric": use_geometric,
                "use_topk": use_topk,
                "n_trials": n_trials,
            },
            f,
            indent=2,
        )

    logger.info(f"Best weights saved to: {results_path}")
    return study.best_params, study.best_value


def main():
    parser = argparse.ArgumentParser(description="precompute and optimize")

    precompute_args = parser.add_argument_group("precompute")
    precompute_args.add_argument(
        "--pipeline", type=str, help="Pipeline configuration to optimize."
    )
    precompute_args.add_argument(
        "--processed-data-dir", type=str, help="Directory containing processed data."
    )
    precompute_args.add_argument("--data-path", type=str, help="path to val datga")
    precompute_args.add_argument("--output-dir", type=str, default="./data/precomputed")
    precompute_args.add_argument("--skip-precompute", action="store_true")

    optimize_args = parser.add_argument_group("Optimize")
    optimize_args.add_argument("--n-trials", type=int, default=50)
    optimize_args.add_argument("--sample-size", type=int, default=10000)
    optimize_args.add_argument("--use-geometric", action="store_true", default=True)
    optimize_args.add_argument("--use-topk", type=int, default=10)

    args = parser.parse_args()

    pipeline_name = args.pipeline

    logger.remove()
    logger.add(sys.stderr, level="INFO", colorize=True)
    logger.add(f"logs/choose_weights_{pipeline_name}.log", level="INFO")

    # Step 1: Precompute hierarchical predictions
    pipeline_config_path = f"{pipeline_name}.yaml"
    hierarchical_cache_path = (
        Path(args.output_dir) / f"{pipeline_name}_hierarchical_validation.pickle"
    )

    if args.skip_precompute and hierarchical_cache_path.exists():
        logger.info(f"Skipping precomputation, using existing cache: {hierarchical_cache_path}")
    else:
        logger.info("Step 1: Precomputing hierarchical predictions...")
        hierarchical_cache_path = run_hierarchical_precomputation(
            pipeline_config_path=pipeline_config_path,
            data_path=args.data_path,
            output_dir=args.output_dir,
            processed_data_dir=args.processed_data_dir,
        )

    # Step 2: Optimize weights
    logger.info("Step 2: Optimizing pipeline weights...")
    validation_data = load_validation_data(Path(args.data_path), args.sample_size)

    best_weights, best_score = optimize_pipeline_weights(
        pipeline_name=pipeline_name,
        hierarchical_cache_path=hierarchical_cache_path,
        validation_data=validation_data,
        processed_data_dir=args.processed_data_dir,
        output_dir=args.output_dir,
        n_trials=args.n_trials,
        use_topk=args.use_topk,
        use_geometric=args.use_geometric,
    )

    logger.info("=" * 50)
    logger.info(f"OPTIMIZATION COMPLETED FOR {pipeline_name}")
    logger.info(f"Best weights: {best_weights}")
    logger.info(f"Best score (top-{args.use_topk}): {best_score:.4f}")
    logger.info("=" * 50)

    print(f"\nOptimization completed for {pipeline_name}")
    print(f"Best weights: {best_weights}")
    print(f"Best score (top-{args.use_topk}): {best_score:.4f}")


if __name__ == "__main__":
    main()
