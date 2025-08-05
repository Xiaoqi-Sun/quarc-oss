import multiprocessing

multiprocessing.set_start_method("spawn", force=True)

import warnings

warnings.filterwarnings("ignore")

import os
import pickle
import torch
import glob
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from typing import List, Dict, Tuple
from functools import partial
from dataclasses import dataclass
from typing import Literal
from loguru import logger
import chemprop
from chemprop import featurizers
from pathlib import Path
import random

from src.quarc.data.gnn_datasets import GNNAgentsDataset
from src.quarc.data.gnn_dataloader import build_dataloader_agent
from src.quarc.models.gnn_models import AgentGNN
from src.quarc.models.modules.gnn_heads import GNNAgentHead
from src.quarc.models.modules.agent_encoder import AgentEncoder
from src.quarc.models.modules.agent_standardizer import AgentStandardizer
from src.quarc.models.search import beam_search_gnn
from eval_utils import get_criteria_fn


@dataclass
class BeamSearchResult:
    doc_id: str
    rxn_class: str
    rxn_smiles: str
    predictions: List[List[int]]  # List of agent indices for each beam
    scores: List[float]  # Corresponding beam scores
    target_indices: List[int]  # Target agent indices


def process_chunk_with_device(
    args,
    model: AgentGNN,
    beam_size: int,
    return_top_n: int,
    a_enc: AgentEncoder,
    a_standardizer: AgentStandardizer,
    featurizer: featurizers.CondensedGraphOfReactionFeaturizer,
):
    chunk, device = args
    torch.cuda.set_device(device)
    return process_chunk(
        chunk_data=chunk,
        model=model,
        a_enc=a_enc,
        a_standardizer=a_standardizer,
        featurizer=featurizer,
        device=device,
        beam_size=beam_size,
        return_top_n=return_top_n,
    )


def process_chunk(
    chunk_data: List,
    model: AgentGNN,
    a_enc: AgentEncoder,
    a_standardizer: AgentStandardizer,
    featurizer: featurizers.CondensedGraphOfReactionFeaturizer,
    device: str,
    beam_size: int = 10,
    return_top_n: int = 10,
) -> List[List[Tuple[torch.Tensor, float]]]:
    """Process a chunk of reactions using beam search"""
    model = model.to(device)
    model.eval()

    dataset = GNNAgentsDataset(
        data=chunk_data,
        agent_standardizer=a_standardizer,
        agent_encoder=a_enc,
        featurizer=featurizer,
    )

    loader = build_dataloader_agent(
        dataset=dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False,
    )

    chunk_results = []

    try:
        with torch.no_grad():
            for idx, (batch) in enumerate(loader):
                a_input, bmg, V_d, X_d, Y, *_ = batch
                a_input = a_input.to(device)
                bmg.to(device)

                beam_results = beam_search_gnn(
                    model=model,
                    bmg=bmg,
                    V_d=V_d,
                    x_d=X_d,
                    num_classes=len(a_enc),
                    agents_input=a_input,
                    max_steps=6,
                    beam_size=beam_size,
                    eos_id=0,
                    return_top_n=return_top_n,
                )

                # Store results as (prediction, score) tuples
                orig_datum = chunk_data[idx]
                chunk_results.append(
                    BeamSearchResult(
                        doc_id=orig_datum.document_id,
                        rxn_class=orig_datum.rxn_class,
                        rxn_smiles=orig_datum.rxn_smiles,
                        predictions=[
                            np.atleast_1d(
                                torch.nonzero(pred.cpu().squeeze()).squeeze().tolist()
                            ).tolist()
                            for pred, _ in beam_results
                        ],
                        scores=[score for _, score in beam_results],
                        target_indices=np.atleast_1d(
                            torch.nonzero(Y.cpu().squeeze()).squeeze().tolist()
                        ).tolist(),
                    )
                )
                torch.cuda.empty_cache()

    finally:
        model = model.cpu()
        del dataset
        del loader
        torch.cuda.empty_cache()

    return chunk_results


def calculate_beam_accuracies(
    results: List[BeamSearchResult],
    criteria: Literal["set", "idx", "combination"] = "set",
    max_k: int = 10,
    a_enc=None,
) -> Dict[int, float]:
    """Calculate top-k accuracies from beam search results."""

    k_hits = {k: 0 for k in range(1, max_k + 1)}
    first_match_idx = []  # 0-based!!

    criteria_fn = get_criteria_fn(criteria)

    for result in results:
        match_found = False

        for i, pred_indices in enumerate(result.predictions):
            k = i + 1  # 1-based indexing bc k_hits is 1-based dict

            if not pred_indices or len(pred_indices) == 0:
                continue

            if criteria_fn(pred_indices, result.target_indices, a_enc):
                first_match_idx.append(i)
                match_found = True

                for hit_k in range(k, max_k + 1):
                    k_hits[hit_k] += 1
                break

        if not match_found:
            first_match_idx.append(-1)

    accuracies = {k: hits / len(results) for k, hits in k_hits.items()}
    return accuracies, first_match_idx


def evaluate_checkpoint(
    checkpoint_path: str,
    test_data: List,
    a_enc: AgentEncoder,
    a_standardizer: AgentStandardizer,
    n_processes: int = 8,
    chunk_size: int = 500,
    gpus: List[str] = None,
    criteria: Literal["set", "idx", "combination"] = "set",
    beam_size: int = 10,
    return_top_n: int = 10,
    depth: int = 2,
    graph_hidden_size: int = 1024,
    n_blocks: int = 3,
    hidden_size: int = 2048,
):
    checkpoint = torch.load(checkpoint_path)

    featurizer = featurizers.CondensedGraphOfReactionFeaturizer(mode_="REAC_DIFF")
    fdims = featurizer.shape

    mp = chemprop.nn.BondMessagePassing(*fdims, depth=depth, d_h=graph_hidden_size)
    agg = chemprop.nn.MeanAggregation()

    predictor = GNNAgentHead(
        graph_input_dim=graph_hidden_size,
        agent_input_dim=len(a_enc),
        output_dim=len(a_enc),
        hidden_dim=hidden_size,
        n_blocks=n_blocks,
    )

    model = AgentGNN(
        message_passing=mp,
        agg=agg,
        predictor=predictor,
        batch_norm=True,
        metrics=[],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    chunks = [test_data[i : i + chunk_size] for i in range(0, len(test_data), chunk_size)]
    pool = Pool(processes=n_processes, maxtasksperchild=1)
    chunk_device_pairs = [(chunk, gpus[i % len(gpus)]) for i, chunk in enumerate(chunks)]
    process_fn = partial(
        process_chunk_with_device,
        model=model,
        a_enc=a_enc,
        a_standardizer=a_standardizer,
        featurizer=featurizer,
        beam_size=beam_size,
        return_top_n=return_top_n,
    )

    all_results = []

    try:
        for chunk_results in tqdm(pool.imap(process_fn, chunk_device_pairs), total=len(chunks)):
            all_results.extend(chunk_results)
    finally:
        pool.close()
        pool.join()

    accuracies, _ = calculate_beam_accuracies(all_results, criteria=criteria, a_enc=a_enc)

    epoch = int(checkpoint_path.split("epoch=")[1].split(".")[0])

    return {
        "epoch": epoch,
        "accuracies": accuracies,
        "checkpoint": checkpoint_path,
    }


def evaluate_multiple_checkpoints(
    checkpoint_dir: str,
    test_data: List,
    a_enc: AgentEncoder,
    a_standardizer: AgentStandardizer,
    featurizer: featurizers.CondensedGraphOfReactionFeaturizer,
    n_processes: int = 8,
    chunk_size: int = 500,
    gpus: List[str] = None,
    criteria: Literal["set", "idx", "combination"] = "set",
    beam_size: int = 10,
    return_top_n: int = 10,
    depth: int = 2,
    graph_hidden_size: int = 1024,
    n_blocks: int = 3,
    hidden_size: int = 2048,
    epochs_to_use: list[int] = None,
):
    """Evaluate multiple checkpoints and return results for comparison."""

    all_checkpoint_paths = glob.glob(os.path.join(checkpoint_dir, "weights-epoch=*.ckpt"))

    if epochs_to_use:
        checkpoint_paths = []
        for epoch in epochs_to_use:
            matching_paths = [
                path
                for path in all_checkpoint_paths
                if int(path.split("epoch=")[1].split(".")[0]) == epoch
            ]
            if matching_paths:
                checkpoint_paths.extend(matching_paths)
            else:
                logger.warning(f"No checkpoint found for epoch {epoch}")
    else:
        checkpoint_paths = all_checkpoint_paths

    checkpoint_paths.sort(key=lambda x: int(x.split("epoch=")[1].split(".")[0]))
    logger.info(f"Found {len(checkpoint_paths)} checkpoints to evaluate")

    results = []
    for cp_path in checkpoint_paths:
        result = evaluate_checkpoint(
            checkpoint_path=cp_path,
            test_data=test_data,
            a_enc=a_enc,
            a_standardizer=a_standardizer,
            n_processes=n_processes,
            chunk_size=chunk_size,
            gpus=gpus,
            criteria=criteria,
            beam_size=beam_size,
            return_top_n=return_top_n,
            depth=depth,
            graph_hidden_size=graph_hidden_size,
            n_blocks=n_blocks,
            hidden_size=hidden_size,
        )
        results.append(result)
        logger.info(
            f"Epoch {result['epoch']} - Top-1: {result['accuracies'][1]:.4f}, Top-5: {result['accuracies'][5]:.4f}, Top-10: {result['accuracies'][10]:.4f}"
        )
    return results


def main(
    checkpoint_dir,
    test_data_path,
    processed_data_dir,
    chunk_size=500,
    n_processes=16,
    epochs_to_use=None,
    depth=2,
    graph_hidden_size=1024,
    n_blocks=3,
    hidden_size=2048,
):
    logger.add("logs/GNN_model_selection_no_rxnclass.log")
    logger.info(f"Evaluating GNN checkpoints (no rxn class) in {checkpoint_dir}")
    logger.info(f"Evaluating test data in {test_data_path}")
    logger.info(f"Using processed data directory: {processed_data_dir}")

    seed = 42
    random.seed(seed)
    logger.info(f"Setting random seed to {seed}, sample 2000 from data")

    processed_data_path = Path(processed_data_dir)
    a_enc = AgentEncoder(class_path=processed_data_path / "agent_encoder/agent_encoder_list.json")
    a_standardizer = AgentStandardizer(
        conv_rules=processed_data_path / "agent_encoder/agent_rules_v1.json",
        other_dict=processed_data_path / "agent_encoder/agent_other_dict.json",
    )
    featurizer = featurizers.CondensedGraphOfReactionFeaturizer(mode_="REAC_DIFF")

    with open(test_data_path, "rb") as f:
        test_data = pickle.load(f)
    test_data = random.sample(test_data, 2000)

    n_gpus = torch.cuda.device_count()
    gpus = [f"cuda:{i}" for i in range(n_gpus)]

    results = evaluate_multiple_checkpoints(
        checkpoint_dir=checkpoint_dir,
        test_data=test_data,
        a_enc=a_enc,
        a_standardizer=a_standardizer,
        featurizer=featurizer,
        n_processes=n_processes,
        chunk_size=chunk_size,
        gpus=gpus,
        criteria="set",
        beam_size=10,
        return_top_n=10,
        depth=depth,
        graph_hidden_size=graph_hidden_size,
        n_blocks=n_blocks,
        hidden_size=hidden_size,
        epochs_to_use=epochs_to_use,
    )

    # Find best checkpoint based on top-5 accuracy
    best_result = max(results, key=lambda x: x["accuracies"][5])
    logger.info("=" * 50)
    logger.info(f"BEST GNN CHECKPOINT (Top-5): {best_result['checkpoint']}")
    logger.info(
        f"Epoch {best_result['epoch']} - Top-1: {best_result['accuracies'][1]:.4f}, Top-5: {best_result['accuracies'][5]:.4f}, Top-10: {best_result['accuracies'][10]:.4f}"
    )

    return best_result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Select best GNN model checkpoint without reaction class"
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, required=True, help="model checkpoint directory"
    )
    parser.add_argument("--test-data-path", type=str, required=True, help="test data file")
    parser.add_argument(
        "--processed-data-dir", type=str, required=True, help="processed data directory"
    )
    parser.add_argument("--chunk-size", type=int, default=500, help="chunk size for evaluation")
    parser.add_argument(
        "--n-processes", type=int, default=16, help="number of processes for evaluation"
    )
    parser.add_argument(
        "--epochs-to-use", type=int, default=None, help="specific epochs to evaluate"
    )
    parser.add_argument("--depth", type=int, default=2, help="depth of message passing")
    parser.add_argument("--graph-hidden-size", type=int, default=1024, help="graph embedding size")
    parser.add_argument("--n-blocks", type=int, default=3, help="number of readout layers")
    parser.add_argument("--hidden-size", type=int, default=2048, help="hidden layer size")

    args = parser.parse_args()

    main(
        checkpoint_dir=args.checkpoint_dir,
        test_data_path=args.test_data_path,
        processed_data_dir=args.processed_data_dir,
        chunk_size=args.chunk_size,
        n_processes=args.n_processes,
        epochs_to_use=args.epochs_to_use,
        depth=args.depth,
        graph_hidden_size=args.graph_hidden_size,
        n_blocks=args.n_blocks,
        hidden_size=args.hidden_size,
    )
