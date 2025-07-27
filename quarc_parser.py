def add_preprocess_opts(parser):
    """Data and path options"""
    group = parser.add_argument_group("quarc_preprocess")

    group.add_argument(
        "--config",
        type=str,
        default="preprocess_config.yaml",
        help="path to preprocess config yaml file",
    )
    group.add_argument(
        "--chunk_json",
        action="store_true",
        help="Run data organization step",
    )
    group.add_argument(
        "--collect_dedup",
        action="store_true",
        help="Run data collection and deduplication",
    )
    group.add_argument(
        "--generate_vocab",
        action="store_true",
        help="Run agent class generation",
    )
    group.add_argument(
        "--init_filter",
        action="store_true",
        help="Run initial filtering",
    )
    group.add_argument(
        "--split",
        action="store_true",
        help="Run train/val/test split",
    )
    group.add_argument(
        "--stage1_filter",
        action="store_true",
        help="Run agent filtering stage",
    )
    group.add_argument(
        "--stage2_filter",
        action="store_true",
        help="Run temperature filtering stage",
    )
    group.add_argument(
        "--stage3_filter",
        action="store_true",
        help="Run agent amount filtering stage",
    )
    group.add_argument(
        "--stage4_filter",
        action="store_true",
        help="Run reactant amount filtering stage",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Run complete pipeline",
    )


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
        "--model-type",
        required=True,
        choices=["ffn", "gnn"],
        help="model architecture",
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

    group.add_argument("--save_dir", type=str, default="./checkpoints", help="checkpoint folder")
    group.add_argument("--logger_name", type=str, help="logger name")
    group.add_argument("--no-cuda", action="store_true", help="use CPU instead of GPU")
    group.add_argument("--gpu", type=int, help="specific GPU device to use")
    group.add_argument("--num-workers", type=int, default=8, help="number of data loading workers")
    group.add_argument("--checkpoint-path", type=str, help="resume from checkpoint")

    # training args
    group.add_argument("--max-epochs", type=int, default=30, help="max num. of epochs")
    group.add_argument("--batch-size", type=int, default=256, help="batch size")
    group.add_argument("--learning-rate", type=float, default=1e-3, help="peak learning rate")
    group.add_argument("--init-lr", type=float, default=1e-4, help="initial learning rate")
    group.add_argument("--final-lr", type=float, default=1e-4, help="final learning rate")
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
    group.add_argument("--config_path", required=True, help="Pipeline config",)
    group.add_argument("--input", "-i", required=True, help="Input JSON file with reactions")
    group.add_argument("--output", "-o", required=True, help="Output JSON file for predictions")
    group.add_argument("--top-k", "-k", type=int, default=10, help="Return top k predictions")


def add_data_opts(parser):
    """Data path options"""
    group = parser.add_argument_group("quarc_data_paths")

    group.add_argument(
        "--processed_data_dir", type=str, default="", help="processed data with encoder files"
    )

    group.add_argument(
        "--train_data_path", type=str, default="", help="stage specific train data path"
    )
    group.add_argument(
        "--val_data_path", type=str, default="", help="stage specific val data path"
    )
    # group.add_argument(
    #     "--test_data_path", type=str, default="", help="stage specific test data path"
    # )
