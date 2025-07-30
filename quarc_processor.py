"""Preprocessing Pipeline for Condition Recommendation

This script implements a multi-stage preprocessing pipeline for reaction data:

1. Data Organization (--chunk_json)
    - Creates initial chunks from raw JSON files
    - Groups chunks for efficient parallel processing

2. Data Collection & Deduplication (--collect_dedup)
    - Extracts key information from reactions (SMILES, temperature, amounts)
    - Deduplicates reactions at condition-level
    - Uses parallel processing with local and global deduplication

3. Initial Filtering (--init_filter)
    - Filters reactions based on basic criteria:
     * Product/reactant/agent count limits
     * Maximum atom counts
     * RDKit parsability

4. Train/Val/Test Split (--split)
    - Splits data by document ID (75:5:20)
    - Ensures related reactions stay together

5. Stage-Specific Filtering (--stage1/2/3/4)
    - Stage 1: Agent existence and amounts
    - Stage 2: Temperature existence and range
    - Stage 3: Reactant amount ratios and uniqueness
    - Stage 4: Agent amount ratios and solvent checks
"""

import yaml
from pathlib import Path
from loguru import logger
from typing import Optional, Dict, Any

from quarc.preprocessing.create_chunks import create_initial_chunks, regroup_chunks
from quarc.preprocessing.collect_deduplicate import collect_and_deduplicate_parallel
from quarc.preprocessing.final_deduplicate import merge_and_deduplicate
from quarc.preprocessing.generate_agent_class import generate_agent_class
from quarc.preprocessing.split_by_document import split_by_document
from quarc.preprocessing.initial_filter import run_initial_filters
from quarc.preprocessing.condition_filter import (
    run_stage1_filters,
    run_stage2_filters,
    run_stage3_filters,
    run_stage4_filters,
)


class QuarcProcessor:
    """QUARC preprocessing pipeline wrapper."""

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._setup_logging()

    def _load_config(self) -> Dict[Any, Any]:
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    def _setup_logging(self):
        log_dir = Path(self.config["logging"]["log_dir"])
        log_dir.mkdir(exist_ok=True)
        logger.remove()

        # Main pipeline logger
        logger.add(
            log_dir / "preprocessing.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {extra[stage]} | {message}",
            level="INFO",
            encoding="utf-8",
            mode="a",
            serialize=False,
        )

        # Stage-specific debug loggers
        if self.config["logging"]["debug_logging"]:
            stages_with_debug = self.config["logging"]["debug_stages"]

            for stage_name, stage_enabled in stages_with_debug.items():
                if stage_enabled:
                    logger.add(
                        log_dir / f"{stage_name}_debug.log",
                        filter=lambda record: record.extra.get("stage") == stage_name,
                        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {extra[stage]} | {message}",
                        level="DEBUG",
                        mode="a",
                        encoding="utf-8",
                        serialize=False,
                    )

    def chunk_data(self):
        """Step 1: Data Organization"""
        with logger.contextualize(stage="chunking"):
            create_initial_chunks(self.config)
            regroup_chunks(self.config)

    def collect_and_deduplicate(self):
        """Step 2: Data Collection & Deduplication"""
        with logger.contextualize(stage="data_collection"):
            collect_and_deduplicate_parallel(self.config)  # collect + local deduplication
            merge_and_deduplicate(self.config)  # final deduplication

    def generate_vocab(self):
        """Step 2.5: Generate agent vocabulary"""
        with logger.contextualize(stage="generate_agent_class"):
            generate_agent_class(self.config)

    def initial_filter(self):
        """Step 3: Initial Filtering"""
        with logger.contextualize(stage="initial_filter"):
            run_initial_filters(self.config)

    def split_data(self):
        """Step 4: Train/Val/Test Split (75:5:20)"""
        with logger.contextualize(stage="split"):
            split_by_document(self.config)

    def stage_filters(self, stages: Optional[list] = None):
        """Step 5: Stage-Specific Filtering"""
        if stages is None:
            stages = [1, 2, 3, 4]

        stage_functions = {
            1: run_stage1_filters,
            2: run_stage2_filters,
            3: run_stage3_filters,
            4: run_stage4_filters,
        }

        for stage in stages:
            if stage in stage_functions:
                with logger.contextualize(stage=f"stage{stage}"):
                    stage_functions[stage](self.config)

    def process_complete_pipeline(self):
        self.chunk_data()
        self.collect_and_deduplicate()
        self.generate_vocab()
        self.initial_filter()
        self.split_data()
        self.stage_filters()

    def process_partial_pipeline(self, steps: list):
        """Run selected steps of the preprocessing pipeline.

        Args:
            steps: List of step names to run. Options:
            ['chunk', 'collect_dedup', 'vocab', 'filter', 'split', 'stage1', 'stage2', 'stage3', 'stage4']
        """
        step_mapping = {
            "chunk": self.chunk_data,
            "collect_dedup": self.collect_and_deduplicate,
            "vocab": self.generate_vocab,
            "filter": self.initial_filter,
            "split": self.split_data,
            "stage1": lambda: self.stage_filters([1]),
            "stage2": lambda: self.stage_filters([2]),
            "stage3": lambda: self.stage_filters([3]),
            "stage4": lambda: self.stage_filters([4]),
        }

        for step in steps:
            if step in step_mapping:
                step_mapping[step]()
            else:
                raise ValueError(f"Unknown step: {step}")


def run_preprocessing(config_path: str, steps: Optional[list] = None, all_steps: bool = False):
    """Main entry point for QUARC preprocessing.

    Args:
        config_path (str): Path to preprocessing configuration YAML
        steps (Optional[list]): List of specific steps to run
        all_steps (bool): If True, run complete pipeline
    """
    processor = QuarcProcessor(config_path)

    if all_steps or steps is None:
        processor.process_complete_pipeline()
    else:
        processor.process_partial_pipeline(steps)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run reaction preprocessing pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        default="preprocess_config.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument("--chunk_json", action="store_true")
    parser.add_argument("--collect_dedup", action="store_true")
    parser.add_argument("--generate_vocab", action="store_true")
    parser.add_argument("--init_filter", action="store_true")
    parser.add_argument("--split", action="store_true")
    parser.add_argument("--stage1_filter", action="store_true")
    parser.add_argument("--stage2_filter", action="store_true")
    parser.add_argument("--stage3_filter", action="store_true")
    parser.add_argument("--stage4_filter", action="store_true")
    parser.add_argument("--all", action="store_true")

    args = parser.parse_args()

    steps = []
    if args.chunk_json:
        steps.append("chunk")
    if args.collect_dedup:
        steps.append("collect_dedup")
    if args.generate_vocab:
        steps.append("vocab")
    if args.init_filter:
        steps.append("filter")
    if args.split:
        steps.append("split")
    if args.stage1_filter:
        steps.append("stage1")
    if args.stage2_filter:
        steps.append("stage2")
    if args.stage3_filter:
        steps.append("stage3")
    if args.stage4_filter:
        steps.append("stage4")

    run_preprocessing(config_path=args.config, steps=steps if steps else None, all_steps=args.all)
