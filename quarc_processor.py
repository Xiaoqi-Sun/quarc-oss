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
import argparse
import os
import sys
from datetime import datetime
import yaml
import re
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
import quarc_parser


def substitute_env_vars(config_dict):
    """Recursively substitute environment variables in config dictionary."""
    if isinstance(config_dict, dict):
        return {k: substitute_env_vars(v) for k, v in config_dict.items()}
    elif isinstance(config_dict, list):
        return [substitute_env_vars(item) for item in config_dict]
    elif isinstance(config_dict, str):
        # Match ${VAR_NAME} pattern and substitute with environment variable
        pattern = r'\$\{([^}]+)\}'
        matches = re.findall(pattern, config_dict)
        result = config_dict
        for match in matches:
            env_value = os.environ.get(match)
            if env_value is None:
                raise EnvironmentError(f"Environment variable {match} is not set")
            result = result.replace(f"${{{match}}}", env_value)
        return result
    else:
        return config_dict


def validate_required_env_vars():
    """Validate that all required environment variables are set."""
    required_vars = [
        "RAW_DIR", "LOG_DIR", "DUMP_DIR", "GROUPED_DIR", "TEMP_DEDUP_DIR", 
        "SPLIT_DIR", "FINAL_DEDUP_PATH", "FINAL_DEDUP_FILTERED_PATH",
        "FINAL_DEDUP_FILTERED_WITH_UNCAT_PATH", "AGENT_ENCODER_LIST_PATH",
        "AGENT_OTHER_DICT_PATH", "CONV_RULES_PATH", "STAGE1_DIR", 
        "STAGE2_DIR", "STAGE3_DIR", "STAGE4_DIR"
    ]
    
    missing_vars = [var for var in required_vars if os.environ.get(var) is None]
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        logger.error("Please set these environment variables or use the Docker script which sets defaults.")
        raise EnvironmentError(f"Missing environment variables: {missing_vars}")


def load_config_with_env_vars(config_path):
    """Load and process configuration file with environment variable substitution."""
    logger.info(f"Loading configuration from {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Substitute environment variables in the config
    try:
        config_with_env = substitute_env_vars(config)
        logger.info("Successfully substituted environment variables in config")
        return config_with_env
    except EnvironmentError as e:
        logger.error(f"Error substituting environment variables: {e}")
        raise

class QuarcProcessor:
    """QUARC preprocessing pipeline wrapper."""

    def __init__(self, config):
        self.config = config

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser("quarc")
    quarc_parser.add_preprocess_opts(parser)
    quarc_parser.add_data_opts(parser)
    args, unknown = parser.parse_known_args()

    # Validate environment variables before loading config
    validate_required_env_vars()
    
    # Load and process config with environment variable substitution
    config = load_config_with_env_vars(args.config)

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

    # create logger
    os.makedirs("./logs/preprocess", exist_ok=True)
    dt = datetime.now().strftime("%y%m%d-%H%Mh")
    log_file = f"./logs/preprocess.{dt}.log"

    logger.remove()
    logger.add(sys.stderr, level="INFO", colorize=True)
    logger.add(log_file, level="INFO")


    processor = QuarcProcessor(config)
    if args.all:
        processor.process_complete_pipeline()
    else:
        processor.process_partial_pipeline(steps)