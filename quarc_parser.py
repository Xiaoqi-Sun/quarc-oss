def add_preprocess_opts(parser):
    """Data and path options"""
    group = parser.add_argument_group("quarc_preprocess")

    group.add_argument("--config", type=str, default="configs/preprocess_config.yaml", help="preprocess config path")
    group.add_argument("--chunk_json", action="store_true", help="run chunk data")
    group.add_argument(
        "--collect_dedup", action="store_true", help="run data collection and deduplication"
    )
    group.add_argument("--generate_vocab", action="store_true", help="run generate vocab")
    group.add_argument("--init_filter", action="store_true", help="run initial filtering")
    group.add_argument("--split", action="store_true", help="run train/val/test split")
    group.add_argument("--stage1_filter", action="store_true", help="run agent filtering")
    group.add_argument("--stage2_filter", action="store_true", help="run temperature filtering")
    group.add_argument(
        "--stage3_filter", action="store_true", help="run reactant amount filtering"
    )
    group.add_argument("--stage4_filter", action="store_true", help="run agent amount filtering")
    group.add_argument("--run_all", action="store_true", help="run complete pipeline")


def add_model_opts(parser):
    """Model architecture options"""
    group = parser.add_argument_group("quarc_model")

    group.add_argument(
        "--stage",
        required=True,
        type=int,
        choices=[1, 2, 3, 4],
        help="training stage, stage 1: agents, stage 2: temperature, stage 3: reactant_amounts, stage 4: agent_amounts",
    )
    group.add_argument(
        "--model-type", required=True, choices=["ffn", "gnn"], help="model architecture"
    )
    group.add_argument("--seed", type=int, default=42, help="random seed")

    # gnn
    group.add_argument("--graph-hidden-size", type=int, default=300, help="graph embedding size")
    group.add_argument("--depth", type=int, default=3, help="number of message passing")
    group.add_argument("--agent-hidden-size", type=int, default=512, help="agent embedding size")

    # ffn
    group.add_argument("--fp-radius", type=int, default=3, help="FP radius")
    group.add_argument("--fp-length", type=int, default=2048, help="FP length")

    # mlp prediction head args
    group.add_argument("--hidden-size", type=int, default=1024, help="hidden layer size")
    group.add_argument("--n-blocks", type=int, default=2, help="number of ffn blocks")
    group.add_argument(
        "--activation",
        choices=["ReLU", "LeakyReLU", "PReLU", "tanh", "SELU", "ELU"],
        default="ReLU",
        help="activation function",
    )
    group.add_argument("--output-size", type=int, required=True, help="mlp readout layer size")
    group.add_argument(
        "--num-classes",
        type=int,
        required=True,
        help="number of agent classes (len(agent_encoder))",
    )


def add_train_opts(parser):
    """Training hyperparameters"""
    group = parser.add_argument_group("quarc_train")

    group.add_argument("--save-dir", type=str, default="./checkpoints", help="checkpoint folder")
    group.add_argument("--logger-name", type=str, help="logger name")
    group.add_argument("--no-cuda", action="store_true", help="use CPU instead of GPU")
    group.add_argument("--gpu", type=int, default=0, help="specific GPU device to use")
    group.add_argument("--num-workers", type=int, default=8, help="number of data loading workers")
    group.add_argument("--checkpoint-path", type=str, default="", help="resume from checkpoint")

    # training args
    group.add_argument("--max-epochs", type=int, default=30, help="max num. of epochs")
    group.add_argument("--batch-size", type=int, default=256, help="batch size per gpu")
    group.add_argument("--max-lr", type=float, default=1e-3, help="peak learning rate")
    group.add_argument("--init-lr", type=float, default=1e-4, help="initial learning rate")
    group.add_argument("--final-lr", type=float, default=1e-4, help="final learning rate")
    group.add_argument("--gamma", type=float, default=0.98, help="gamma for exponential LR")
    group
    group.add_argument(
        "--warmup-epochs",
        type=float,
        default=2.0,
        help="number of warmup epochs",
    )

    group.add_argument("--early-stop", action="store_true", help="enable early stopping")
    group.add_argument(
        "--early-stop-patience",
        type=int,
        default=5,
        help="early stopping patience",
    )


def add_predict_opts(parser):
    """Predicting options"""
    group = parser.add_argument_group("quarc_predict")
    group.add_argument(
        "--config-path",
        type=str,
        default="configs/hybrid_pipeline_oss.yaml",
        help="Pipeline config",
    )
    group.add_argument(
        "--input", "-i", type=str, default="", help="Input JSON file with reactions"
    )
    group.add_argument(
        "--output", "-o", type=str, default="", help="Output JSON file for predictions"
    )
    group.add_argument("--top-k", type=int, default=10, help="top k predictions")


def add_data_opts(parser):
    """Data path options"""
    group = parser.add_argument_group("quarc_data_paths")

    group.add_argument(
        "--processed-data-dir",
        type=str,
        default="./data/processed",
        help="processed data with encoder files",
    )

    group.add_argument(
        "--train-data-path", type=str, default="", help="stage specific train data path"
    )
    group.add_argument(
        "--val-data-path", type=str, default="", help="stage specific val data path"
    )


def add_server_opts(parser):
    group = parser.add_argument_group("quarc_server")

    group.add_argument("--server-ip", help="Server IP to use", type=str, default="0.0.0.0")
    group.add_argument("--server-port", help="Server port to use", type=int, default=9910)
