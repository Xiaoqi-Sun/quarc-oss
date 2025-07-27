import argparse
import os
import sys
import pickle
import time
import torch
import yaml
from pathlib import Path
from typing import Tuple, List, Any

import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from torch.utils.data import DataLoader
from loguru import logger

import quarc_parser

class ModelFactory:
    """Factory for creating models and datasets based on argparse arguments"""

    def __init__(self, args):
        self.args = args

        self._load_encoders()

    def _load_encoders(self):
        """Load agent encoder and standardizer"""
        from quarc.models.modules.agent_encoder import AgentEncoder
        from quarc.models.modules.agent_standardizer import AgentStandardizer

        self.agent_encoder = AgentEncoder(
            class_path=self.args.processed_data_dir / "agent_encoder/agent_encoder_list.json"
        )
        self.agent_standardizer = AgentStandardizer(
            conv_rules=self.args.processed_data_dir / "agent_encoder/agent_rules_v1.json",
            other_dict=self.args.processed_data_dir / "agent_encoder/agent_other_dict.json",
        )

    def create_model_and_data(
        self, train_data, val_data
    ) -> Tuple[pl.LightningModule, DataLoader, DataLoader, List]:
        """Create model and data loaders for the specified stage"""

        if self.args.model_type == "ffn":
            return self._create_ffn_model_and_data(train_data, val_data)
        elif self.args.model_type == "gnn":
            return self._create_gnn_model_and_data(train_data, val_data)
        else:
            raise ValueError(f"Unknown model_type: {self.args.model_type}")

    def _create_ffn_model_and_data(self, train_data, val_data):
        """Create FFN model and datasets for current stage"""
        stage = self.args.stage

        if stage == 1:
            return self._create_ffn_stage1(train_data, val_data)
        elif stage == 2:
            return self._create_ffn_stage2(train_data, val_data)
        elif stage == 3:
            return self._create_ffn_stage3(train_data, val_data)
        elif stage == 4:
            return self._create_ffn_stage4(train_data, val_data)
        else:
            raise ValueError(f"Invalid stage: {stage}")

    def _create_ffn_stage1(self, train_data, val_data):
        """FFN Stage 1: Agent prediction"""
        from quarc.data.ffn_datasets import AugmentedAgentsDataset, AgentsDataset
        from quarc.models.ffn_models import AgentFFN
        from quarc.models.modules.ffn_heads import FFNAgentHead
        from quarc.models.callbacks import FFNGreedySearchCallback
        from torcheval.metrics.functional import multilabel_accuracy

        # Datasets
        train_dataset = AugmentedAgentsDataset(
            original_data=train_data,
            agent_standardizer=self.agent_standardizer,
            agent_encoder=self.agent_encoder,
            sample_weighting="pascal",
            fp_radius=self.args.fp_radius,
            fp_length=self.args.fp_length,
        )
        val_dataset = AgentsDataset(
            data=val_data,
            agent_standardizer=self.agent_standardizer,
            agent_encoder=self.agent_encoder,
        )

        # Data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
        )

        # Model
        predictor = FFNAgentHead(
            fp_dim=self.args.fp_length,
            agent_input_dim=len(self.agent_encoder),
            output_dim=self.args.output_size,
            hidden_dim=self.args.hidden_size,
            n_blocks=self.args.n_blocks,
        )

        metrics = {
            "multilabel_accuracy_exactmatch": multilabel_accuracy,
            "multilabel_accuracy_hamming": lambda preds, targets: multilabel_accuracy(
                preds, targets, criteria="hamming"
            ),
        }

        model = AgentFFN(
            predictor=predictor,
            metrics=metrics,
            init_lr=self.args.init_lr,
            max_lr=self.args.max_lr,
            final_lr=self.args.final_lr,
        )

        # Special callback for stage 1
        greedy_callback = FFNGreedySearchCallback(track_batch_indices=range(len(val_loader)))
        extra_callbacks = [greedy_callback]

        return model, train_loader, val_loader, extra_callbacks

    def _create_ffn_stage2(self, train_data, val_data):
        """FFN Stage 2: Temperature prediction"""
        from quarc.data.ffn_datasets import BinnedTemperatureDataset
        from quarc.models.ffn_models import TemperatureFFN
        from quarc.models.modules.ffn_heads import FFNTemperatureHead
        from torchmetrics.classification import Accuracy
        from rdkit import Chem

        # Fingerprint generator
        fp_gen = Chem.rdFingerprintGenerator.GetMorganGenerator(
            radius=self.args.fp_radius, fpSize=self.args.fp_length
        )

        # Datasets
        train_dataset = BinnedTemperatureDataset(
            data=train_data,
            morgan_generator=fp_gen,
            agent_standardizer=self.agent_standardizer,
            agent_encoder=self.agent_encoder,
        )
        val_dataset = BinnedTemperatureDataset(
            data=val_data,
            morgan_generator=fp_gen,
            agent_standardizer=self.agent_standardizer,
            agent_encoder=self.agent_encoder,
        )

        # Data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
        )

        # Model
        predictor = FFNTemperatureHead(
            fp_dim=self.args.fp_length,
            agent_input_dim=len(self.agent_encoder),
            output_dim=self.args.output_size,
            hidden_dim=self.args.hidden_size,
            n_blocks=self.args.n_blocks,
        )

        metrics = {
            "accuracy": Accuracy(
                task="multiclass",
                average="micro",
                num_classes=self.args.output_size,
                ignore_index=0,
            ),
        }

        model = TemperatureFFN(
            predictor=predictor,
            metrics=metrics,
            warmup_epochs=self.args.warmup_epochs,
            init_lr=self.args.init_lr,
            max_lr=self.args.max_lr,
            final_lr=self.args.final_lr,
        )

        return model, train_loader, val_loader, []

    def _create_ffn_stage3(self, train_data, val_data):
        """FFN Stage 3: Reactant amount prediction"""
        from quarc.data.ffn_datasets import BinnedReactantAmountDataset
        from quarc.models.ffn_models import ReactantAmountFFN
        from quarc.models.modules.ffn_heads import FFNReactantAmountHead
        from torchmetrics.classification import Accuracy

        # Datasets
        train_dataset = BinnedReactantAmountDataset(
            data=train_data,
            agent_standardizer=self.agent_standardizer,
            agent_encoder=self.agent_encoder,
            fp_radius=self.args.fp_radius,
            fp_length=self.args.fp_length,
        )
        val_dataset = BinnedReactantAmountDataset(
            data=val_data,
            agent_standardizer=self.agent_standardizer,
            agent_encoder=self.agent_encoder,
            fp_radius=self.args.fp_radius,
            fp_length=self.args.fp_length,
        )

        # Data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
        )

        # Model
        predictor = FFNReactantAmountHead(
            fp_dim=self.args.fp_length,
            agent_input_dim=len(self.agent_encoder),
            output_dim=self.args.output_size,
            hidden_dim=self.args.hidden_size,
            n_blocks=self.args.n_blocks,
        )

        metrics = {
            "accuracy": Accuracy(
                task="multiclass",
                average="micro",
                num_classes=self.args.output_size,
                ignore_index=0,
            ),
        }

        model = ReactantAmountFFN(
            predictor=predictor,
            metrics=metrics,
            warmup_epochs=self.args.warmup_epochs,
            init_lr=self.args.init_lr,
            max_lr=self.args.max_lr,
            final_lr=self.args.final_lr,
        )

        return model, train_loader, val_loader, []

    def _create_ffn_stage4(self, train_data, val_data):
        """FFN Stage 4: Agent amount prediction"""
        from quarc.data.ffn_datasets import BinnedAgentAmoutOneshot
        from quarc.models.ffn_models import AgentAmountFFN
        from quarc.models.modules.ffn_heads import FFNAgentAmountHead
        from torchmetrics.classification import Accuracy
        from rdkit import Chem

        # Fingerprint generator
        fp_gen = Chem.rdFingerprintGenerator.GetMorganGenerator(
            radius=self.args.fp_radius, fpSize=self.args.fp_length
        )

        # Datasets
        train_dataset = BinnedAgentAmoutOneshot(
            data=train_data,
            morgan_generator=fp_gen,
            agent_standardizer=self.agent_standardizer,
            agent_encoder=self.agent_encoder,
        )
        val_dataset = BinnedAgentAmoutOneshot(
            data=val_data,
            morgan_generator=fp_gen,
            agent_standardizer=self.agent_standardizer,
            agent_encoder=self.agent_encoder,
        )

        # Data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
        )

        # Model
        predictor = FFNAgentAmountHead(
            fp_dim=self.args.fp_length,
            agent_input_dim=len(self.agent_encoder),
            output_dim=self.args.output_size,
            hidden_dim=self.args.hidden_size,
            n_blocks=self.args.n_blocks,
        )

        metrics = {
            "accuracy": Accuracy(
                task="multiclass",
                average="micro",
                num_classes=self.args.output_size,
                ignore_index=0,
            ),
        }

        model = AgentAmountFFN(
            predictor=predictor,
            metrics=metrics,
            warmup_epochs=self.args.warmup_epochs,
            init_lr=self.args.init_lr,
            max_lr=self.args.max_lr,
            final_lr=self.args.final_lr,
        )

        return model, train_loader, val_loader, []

    def _create_gnn_model_and_data(self, train_data, val_data):
        """Create GNN model and datasets for current stage"""
        stage = self.args.stage

        if stage == 1:
            return self._create_gnn_stage1(train_data, val_data)
        elif stage == 2:
            return self._create_gnn_stage2(train_data, val_data)
        elif stage == 3:
            return self._create_gnn_stage3(train_data, val_data)
        elif stage == 4:
            return self._create_gnn_stage4(train_data, val_data)
        else:
            raise ValueError(f"Invalid stage: {stage}")

    def _create_gnn_stage1(self, train_data, val_data):
        """GNN Stage 1: Agent prediction"""
        import chemprop
        from chemprop import featurizers
        from quarc.data.gnn_datasets import GNNAugmentedAgentsDataset, GNNAgentsDataset
        from quarc.models.gnn_models import AgentGNN
        from quarc.models.modules.gnn_heads import GNNAgentHead
        from quarc.models.callbacks import GNNGreedySearchCallback
        from quarc.data.gnn_dataloader import build_dataloader_agent
        from torcheval.metrics.functional import multilabel_accuracy

        # Featurizer
        featurizer = featurizers.CondensedGraphOfReactionFeaturizer(mode_="REAC_DIFF")

        # Datasets
        train_dataset = GNNAugmentedAgentsDataset(
            original_data=train_data,
            agent_standardizer=self.agent_standardizer,
            agent_encoder=self.agent_encoder,
            featurizer=featurizer,
        )
        val_dataset = GNNAgentsDataset(
            data=val_data,
            agent_standardizer=self.agent_standardizer,
            agent_encoder=self.agent_encoder,
            featurizer=featurizer,
        )

        # Data loaders
        train_loader = build_dataloader_agent(
            dataset=train_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=True,
            distributed=False,
            persistent_workers=False,
            pin_memory=True,
        )
        val_loader = build_dataloader_agent(
            dataset=val_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=False,
            distributed=False,
            persistent_workers=False,
            pin_memory=True,
        )

        # Model
        fdims = featurizer.shape
        mp = chemprop.nn.BondMessagePassing(
            *fdims, d_h=self.args.graph_hidden_size, depth=self.args.depth
        )
        agg = chemprop.nn.MeanAggregation()

        predictor = GNNAgentHead(
            graph_input_dim=self.args.graph_hidden_size,
            agent_input_dim=len(self.agent_encoder),
            output_dim=len(self.agent_encoder),
            hidden_dim=self.args.hidden_size,
            n_blocks=self.args.n_blocks,
        )

        metrics = {
            "multilabel_accuracy_exactmatch": multilabel_accuracy,
            "multilabel_accuracy_hamming": lambda preds, targets: multilabel_accuracy(
                preds, targets, criteria="hamming"
            ),
        }

        model = AgentGNN(
            message_passing=mp,
            agg=agg,
            predictor=predictor,
            batch_norm=True,
            metrics=metrics,
            init_lr=self.args.init_lr,
        )

        # Special callback for stage 1
        greedy_callback = GNNGreedySearchCallback(track_batch_indices=range(len(val_loader)))
        extra_callbacks = [greedy_callback]

        return model, train_loader, val_loader, extra_callbacks

    def _create_gnn_stage2(self, train_data, val_data):
        """GNN Stage 2: Temperature prediction"""
        import chemprop
        from chemprop import featurizers
        from quarc.data.gnn_datasets import GNNBinnedTemperatureDataset
        from quarc.models.gnn_models import TemperatureGNN
        from quarc.models.modules.gnn_heads import GNNTemperatureHead
        from quarc.data.gnn_dataloader import build_dataloader_agent
        from torchmetrics.classification import Accuracy

        # Featurizer
        featurizer = featurizers.CondensedGraphOfReactionFeaturizer(mode_="REAC_DIFF")

        # Datasets
        train_dataset = GNNBinnedTemperatureDataset(
            data=train_data,
            agent_standardizer=self.agent_standardizer,
            agent_encoder=self.agent_encoder,
            featurizer=featurizer,
        )
        val_dataset = GNNBinnedTemperatureDataset(
            data=val_data,
            agent_standardizer=self.agent_standardizer,
            agent_encoder=self.agent_encoder,
            featurizer=featurizer,
        )

        # Data loaders
        train_loader = build_dataloader_agent(
            dataset=train_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=True,
            classification=True,
            distributed=False,
            persistent_workers=True,
            pin_memory=True,
        )
        val_loader = build_dataloader_agent(
            dataset=val_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=False,
            classification=True,
            distributed=False,
            persistent_workers=True,
            pin_memory=True,
        )

        # Model
        fdims = featurizer.shape
        mp = chemprop.nn.BondMessagePassing(
            *fdims, d_h=self.args.graph_hidden_size, depth=self.args.depth
        )
        agg = chemprop.nn.MeanAggregation()

        predictor = GNNTemperatureHead(
            graph_input_dim=self.args.graph_hidden_size,
            agent_input_dim=len(self.agent_encoder),
            output_dim=self.args.output_size,
            hidden_dim=self.args.hidden_size,
            n_blocks=self.args.n_blocks,
        )

        metrics = {
            "accuracy": Accuracy(
                task="multiclass",
                average="micro",
                num_classes=self.args.output_size,
                ignore_index=0,
            ),
        }

        model = TemperatureGNN(
            message_passing=mp,
            agg=agg,
            predictor=predictor,
            batch_norm=True,
            metrics=metrics,
            warmup_epochs=self.args.warmup_epochs,
            init_lr=self.args.init_lr,
            max_lr=self.args.max_lr,
            final_lr=self.args.final_lr,
        )

        return model, train_loader, val_loader, []

    def _create_gnn_stage3(self, train_data, val_data):
        """GNN Stage 3: Reactant amount prediction"""
        import chemprop
        from chemprop import featurizers
        from quarc.data.gnn_datasets import GNNBinnedReactantAmountDataset
        from quarc.models.gnn_models import ReactantAmountGNN
        from quarc.models.modules.gnn_heads import GNNReactantAmountHead
        from quarc.data.gnn_dataloader import build_dataloader_agent
        from torchmetrics.classification import Accuracy
        from rdkit import Chem

        # Featurizer
        featurizer = featurizers.CondensedGraphOfReactionFeaturizer(mode_="REAC_DIFF")
        fp_gen = Chem.rdFingerprintGenerator.GetMorganGenerator(
            radius=self.args.fp_radius, fpSize=self.args.fp_length
        )

        # Datasets
        train_dataset = GNNBinnedReactantAmountDataset(
            data=train_data,
            agent_standardizer=self.agent_standardizer,
            agent_encoder=self.agent_encoder,
            featurizer=featurizer,
            morgan_generator=fp_gen,
        )
        val_dataset = GNNBinnedReactantAmountDataset(
            data=val_data,
            agent_standardizer=self.agent_standardizer,
            agent_encoder=self.agent_encoder,
            featurizer=featurizer,
            morgan_generator=fp_gen,
        )

        # Data loaders
        train_loader = build_dataloader_agent(
            dataset=train_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=True,
            classification=True,
            distributed=False,
            persistent_workers=True,
            pin_memory=True,
        )
        val_loader = build_dataloader_agent(
            dataset=val_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=False,
            classification=True,
            distributed=False,
            persistent_workers=True,
            pin_memory=True,
        )

        # Model
        fdims = featurizer.shape
        mp = chemprop.nn.BondMessagePassing(
            *fdims, d_h=self.args.graph_hidden_size, depth=self.args.depth
        )
        agg = chemprop.nn.MeanAggregation()

        predictor = GNNReactantAmountHead(
            graph_input_dim=self.args.graph_hidden_size,
            agent_input_dim=len(self.agent_encoder),
            output_dim=self.args.output_size,
            hidden_dim=self.args.hidden_size,
            n_blocks=self.args.n_blocks,
        )

        metrics = {
            "accuracy": Accuracy(
                task="multiclass",
                average="micro",
                num_classes=self.args.output_size,
                ignore_index=0,
            ),
        }

        model = ReactantAmountGNN(
            message_passing=mp,
            agg=agg,
            predictor=predictor,
            batch_norm=True,
            metrics=metrics,
            warmup_epochs=self.args.warmup_epochs,
            init_lr=self.args.init_lr,
            max_lr=self.args.max_lr,
            final_lr=self.args.final_lr,
        )

        return model, train_loader, val_loader, []

    def _create_gnn_stage4(self, train_data, val_data):
        """GNN Stage 4: Agent amount prediction"""
        import chemprop
        from chemprop import featurizers
        from quarc.data.gnn_datasets import GNNBinnedAgentAmountOneShotDataset
        from quarc.models.gnn_models import AgentAmountOneshotGNN
        from quarc.models.modules.gnn_heads import GNNAgentAmountHead
        from quarc.data.gnn_dataloader import build_dataloader_agent
        from torchmetrics.classification import Accuracy

        # Featurizer
        featurizer = featurizers.CondensedGraphOfReactionFeaturizer(mode_="REAC_DIFF")

        # Datasets
        train_dataset = GNNBinnedAgentAmountOneShotDataset(
            data=train_data,
            agent_standardizer=self.agent_standardizer,
            agent_encoder=self.agent_encoder,
            featurizer=featurizer,
        )
        val_dataset = GNNBinnedAgentAmountOneShotDataset(
            data=val_data,
            agent_standardizer=self.agent_standardizer,
            agent_encoder=self.agent_encoder,
            featurizer=featurizer,
        )

        # Data loaders
        train_loader = build_dataloader_agent(
            dataset=train_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=True,
            classification=True,
            distributed=False,
            persistent_workers=True,
            pin_memory=True,
        )
        val_loader = build_dataloader_agent(
            dataset=val_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=False,
            classification=True,
            distributed=False,
            persistent_workers=True,
            pin_memory=True,
        )

        # Model
        fdims = featurizer.shape
        mp = chemprop.nn.BondMessagePassing(
            *fdims, d_h=self.args.graph_hidden_size, depth=self.args.depth
        )
        agg = chemprop.nn.MeanAggregation()

        predictor = GNNAgentAmountHead(
            graph_dim=self.args.graph_hidden_size,
            agent_input_dim=len(self.agent_encoder),
            output_dim=self.args.output_size,
            hidden_dim=self.args.hidden_size,
            n_blocks=self.args.n_blocks,
        )

        metrics = {
            "accuracy": Accuracy(
                task="multiclass",
                average="micro",
                num_classes=self.args.output_size,
                ignore_index=0,
            ),
        }

        model = AgentAmountOneshotGNN(
            message_passing=mp,
            agg=agg,
            predictor=predictor,
            batch_norm=True,
            metrics=metrics,
            warmup_epochs=self.args.warmup_epochs,
            init_lr=self.args.init_lr,
            max_lr=self.args.max_lr,
            final_lr=self.args.final_lr,
        )

        return model, train_loader, val_loader, []


class QuarcTrainer:
    """Single-stage Trainer for QUARC models"""

    def __init__(self, args):
        self.args = args

        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))

        pl.seed_everything(self.args.seed, workers=True)

        self.model_factory = ModelFactory(self.args)

        logger.info(f"Initialized QUARC trainer: {self.args.model_type} Stage {self.args.stage}")

    def _setup_distributed(self):
        if self.world_size > 1 and not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo",
                world_size=self.world_size,
                rank=self.local_rank,
            )
        if torch.cuda.is_available() and not self.args.no_cuda:
            torch.cuda.set_device(self.local_rank)

    def _setup_tblogger(self):
        save_dir = Path("./models") / self.args.model_type.upper() / f"stage{self.args.stage}"
        save_dir.mkdir(parents=True, exist_ok=True)

        tb_logger = pl_loggers.TensorBoardLogger(
            save_dir=save_dir,
            name=self.args.logger_name,
        )

        if self.local_rank == 0:
            self._save_args(tb_logger.log_dir)

        return tb_logger

    @rank_zero_only
    def _save_args(self, log_dir):
        args_dict = vars(self.args)
        args_yaml = yaml.dump(args_dict, indent=2, default_flow_style=False)

        os.makedirs(log_dir, exist_ok=True)
        with open(f"{log_dir}/args.yaml", "w") as f:
            f.write(args_yaml)

        logger.info(f"Training arguments:\n{args_yaml}")

    def _load_stage_data(self):
        """Load training and validation data for current stage"""
        train_data_path = self.args.train_data_path
        val_data_path = self.args.val_data_path

        with open(train_data_path, "rb") as f:
            train_data = pickle.load(f)
        with open(val_data_path, "rb") as f:
            val_data = pickle.load(f)

        if self.local_rank == 0:
            logger.info(f"Stage {self.args.stage} data: train={len(train_data)}, val={len(val_data)}")

        return train_data, val_data

    def _setup_callbacks(self, tb_logger, extra_callbacks=None):
        stage = self.args.stage
        tb_path = tb_logger.log_dir

        weights_checkpoint = ModelCheckpoint(
            dirpath=tb_path,
            filename="weights-{epoch}",
            save_weights_only=True,
            save_last=False,
            every_n_epochs=1 if stage == 1 else 5,
            save_top_k=-1,
        )

        full_checkpoint = ModelCheckpoint(
            dirpath=tb_path,
            filename="full-{epoch}",
            save_weights_only=False,
            save_last=False,
            every_n_epochs=3 if stage == 1 else 10,
            save_top_k=-1,
        )

        if stage == 1:
            earlystop_callback = EarlyStopping(
                monitor="val_greedy_exactmatch_accuracy",
                patience=5,
                mode="max",
                check_on_train_epoch_end=False,
            )
        else:
            earlystop_callback = EarlyStopping(monitor="accuracy", patience=15, mode="max")

        callbacks = [weights_checkpoint, full_checkpoint, earlystop_callback]

        if extra_callbacks:
            callbacks.extend(extra_callbacks)

        return callbacks

    def train(self) -> str:
        """
        Train the model for the current stage

        Returns:
            Path to trained model checkpoint directory
        """
        logger.info(f"Starting {self.args.model_type.upper()} Stage {self.args.stage} training")

        self._setup_distributed()
        tb_logger = self._setup_tblogger()

        # Load data and model
        train_data, val_data = self._load_stage_data()
        model, train_loader, val_loader, extra_callbacks = (
            self.model_factory.create_model_and_data(train_data, val_data)
        )

        callbacks = self._setup_callbacks(tb_logger, extra_callbacks)

        trainer = pl.Trainer(
            logger=tb_logger,
            accelerator="cpu" if self.args.no_cuda else "gpu",
            devices="auto",
            strategy="ddp" if self.world_size > 1 else "auto",
            callbacks=callbacks,
            max_epochs=self.args.max_epochs,
            sync_batchnorm=True,
            use_distributed_sampler=True if self.world_size > 1 else False,
            deterministic=True,
        )

        # Train
        try:
            if self.args.checkpoint_path:
                trainer.fit(model, train_loader, val_loader, ckpt_path=self.args.checkpoint_path)
            else:
                trainer.fit(model, train_loader, val_loader)


        except Exception as e:
            logger.error(f"Training failed for {self.args.model_type} stage {self.args.stage}: {e}")
            raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser("quarc")
    quarc_parser.add_model_opts(parser)
    quarc_parser.add_train_opts(parser)
    quarc_parser.add_data_opts(parser)

    args, unknown = parser.parse_known_args()

    # Create logger
    logger.remove()
    logger.add(sys.stderr, level="INFO", colorize=True)
    os.makedirs("./logs/train", exist_ok=True)
    log_file = f"./logs/train/{args.logger_name}/{args.model_type}_stage{args.stage}.log"
    logger.add(log_file, level="INFO")

    # Train the stage
    trainer = QuarcTrainer(args)
    trainer.train()
