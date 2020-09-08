# IPDB can be used for debugging.
# Ignoring flake8 error code F401
import ipdb  # noqa: F401
import argparse
import os

from pathlib import Path
import tensorflow as tf
import tensorflow_addons as tfa

from detr_models.data_feeder.coco_feeder import COCOFeeder
from detr_models.data_feeder.pvoc_feeder import PVOCFeeder
from detr_models.detr.config import DefaultDETRConfig
from detr_models.detr.model import DETR
from detr_models.detr.utils import get_decay_schedules, get_image_information

tf.keras.backend.set_floatx("float32")


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
        "-tm",
        "--train_masks",
        help="Flag to indicate trainining of segmentation masks",
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


def init_training(training_config: dict):
    """Initialize DETR training procedure

    Parameters
    ----------
    training_config : dictionary
        Configuration used for training.
    """
    if training_config["use_gpu"]:
        assert tf.config.list_physical_devices("GPU"), "No GPU available"
        assert tf.test.is_built_with_cuda(), "Tensorflow not compiled with CUDA support"

    if config.data_type not in ["PVOC", "COCO"]:
        raise ValueError(
            "Invalid `data_type` specified in Config: {}".format(config.data_type)
        )

    # Get image input shape and number of images in path
    input_shape, count_images = get_image_information(
        training_config["storage_path"], config.data_type
    )

    # Init Backbone Config
    backbone_config = {
        "input_shape": input_shape,
        "include_top": False,
        "weights": "imagenet",
    }

    detr = DETR(
        input_shape=input_shape,
        num_queries=config.num_queries,
        num_classes=config.num_classes,
        num_heads=config.num_heads,
        dim_transformer=config.dim_transformer,
        dim_feedforward=config.dim_feedforward,
        num_transformer_layer=config.num_transformer_layer,
        backbone_name=config.backbone_name,
        backbone_config=backbone_config,
        train_backbone=config.train_backbone,
    )

    detr.build_model(training_config["use_pretrained"], training_config["train_masks"])

    # Initialize data feeder
    if config.data_type == "PVOC":
        print("Use Input Data in PascalVOC format")
        data_feeder = PVOCFeeder(
            storage_path=training_config["storage_path"],
            batch_size=training_config["batch_size"],
            num_queries=config.num_queries,
            num_classes=config.num_classes,
        )
    else:
        print("Use Input Data in COCO format")
        data_feeder = COCOFeeder(
            storage_path=training_config["storage_path"],
            batch_size=training_config["batch_size"],
            num_queries=config.num_queries,
            num_classes=config.num_classes,
            image_width=config.image_width,
            image_height=config.image_height,
        )

    lr_schedule, wd_schedule = get_decay_schedules(
        num_steps=count_images // training_config["batch_size"],
        lr=config.learning_rate,
        drops=config.drops,
        weight_decay=config.weight_decay,
    )

    optimizer = tfa.optimizers.AdamW(
        weight_decay=wd_schedule, learning_rate=lr_schedule
    )

    detr.train(
        training_config=training_config,
        optimizer=optimizer,
        count_images=count_images,
        data_feeder=data_feeder,
    )


if __name__ == "__main__":
    config = DefaultDETRConfig()

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
        "use_gpu": args.use_gpu,
        "verbose": args.verbose,
    }

    init_training(training_config)
