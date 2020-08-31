import argparse
import os
from pathlib import Path

from detr_models.detr.config import DefaultDETRConfig
from detr_models.detr.train import init_training

config = DefaultDETRConfig()


def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "-sp", "--storage_path", help="Path to data storage", type=str, required=True
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        help="Path to store weights and losses",
        default=os.path.join(os.getcwd(), "training_results"),
        type=str,
    )

    parser.add_argument("-bs", "--batch_size", default=config.batch_size, type=int)
    parser.add_argument("-e", "--epochs", default=config.epochs, type=int)

    parser.add_argument(
        "-masks",
        "--train_masks",
        help="Flag to indicate trainining of segmentation masks",
        default=config.train_masks,
        action="store_true",
    )

    parser.add_argument(
        "-up", "--use_pretrained", help="Path to pre-trained model weights.", type=str
    )

    parser.add_argument(
        "-gpu",
        "--use_gpu",
        help="Flag to indicate training on a GPU",
        action="store_true",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        help="Flag to indicate additional logging",
        action="store_true",
    )
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "DETR training script", parents=[get_args_parser()]
    )
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    training_config = {
        "storage_path": args.storage_path,
        "output_dir": args.output_dir,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "train_masks": args.train_masks,
        "use_pretrained": args.use_pretrained,
        "use_gpu": args.use_gpu,
        "verbose": args.verbose,
    }

    init_training(training_config)
