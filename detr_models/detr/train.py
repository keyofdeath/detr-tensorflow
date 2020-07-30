import argparse
import os
from pathlib import Path

# IPDB can be used for debugging.
# Ignoring flake8 error code F401
import ipdb  # noqa: F401
import tensorflow as tf
import tensorflow_addons as tfa
from detr_models.detr.config import DefaultDETRConfig
from detr_models.detr.model import DETR
from tensorflow.keras.preprocessing.image import img_to_array, load_img

tf.keras.backend.set_floatx("float32")
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

    parser.add_argument(
        "-lr", "--learning_rate", default=config.learning_rate, type=float
    )
    parser.add_argument(
        "-wd", "--weight_decay", default=config.weight_decay, type=float
    )
    parser.add_argument(
        "-d",
        "--drops",
        help="Learning rate and weight decay drop after epochs.",
        nargs="+",
        default=config.drops,
    )

    parser.add_argument("-bs", "--batch_size", default=config.batch_size, type=int)
    parser.add_argument("-e", "--epochs", default=config.epochs, type=int)

    parser.add_argument(
        "-up", "--use_pretrained", help="Path to pre-trained model weights.", type=str
    )

    parser.add_argument(
        "-nq",
        "--num_queries",
        help="Number of queries used in transformer",
        default=config.num_queries,
        type=int,
    )
    parser.add_argument(
        "-nc",
        "--num_classes",
        help="Number of classes in classification",
        default=config.num_classes,
        type=int,
    )
    parser.add_argument(
        "-nh",
        "--num_heads",
        help="Number of heads in transformer",
        default=config.num_heads,
        type=int,
    )
    parser.add_argument(
        "-ntl",
        "--num_transformer_layer",
        help="Number of transformer layers",
        default=config.num_transformer_layer,
        type=int,
    )

    parser.add_argument(
        "-dt",
        "--dim_transformer",
        help="Number of transformer units",
        default=config.dim_transformer,
        type=int,
    )
    parser.add_argument(
        "-df",
        "--dim_feedforward",
        help="Number of feed forwards neurons in transformer",
        default=config.dim_feedforward,
        type=int,
    )

    parser.add_argument(
        "-bn",
        "--backbone_name",
        help="Name of backbone to use",
        default=config.backbone_name,
        type=str,
        choices=["ResNet50", "MobileNetV2", "InceptionV3"],
    )
    parser.add_argument(
        "-tb",
        "--train_backbone",
        help="Flag to indicate training of backbone",
        action="store_true",
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
        default=False,
        action="store_true",
    )
    return parser


def get_image_information(storage_path: str):
    """Helper function to retrieve image information.

    Parameters
    ----------
    storage_path : str
        Path to data storage

    Returns
    -------
    input_shape : tuple
        Input shape of images [H, W, C]
    count_images : int
        Number of images stored in `storage_path`
    """

    image_path = "{}/{}".format(storage_path, "images")
    images = os.listdir(image_path)
    count_images = len(images)

    sample_image = img_to_array(load_img("{}/{}".format(image_path, images[0])))
    input_shape = sample_image.shape
    return input_shape, count_images


def get_decay_schedules(num_steps: int, lr: float, drops: list, weight_decay: float):
    """Helper function to create learning rate and weight decay schedules.

    Parameters
    ----------
    num_steps : int
        Number of training steps per epoch.
    lr : float
        Learning rate at beginning.
    drops : list
        Epochs after which lr and wd should drop.
    weight_decay : float
        Weight decay multiplier.

    Returns
    -------
    tf.optimizer.schedules, tf.optimizer.schedules
        Learning rate and weight decay schedules.
    """
    boundaries = [drop * num_steps for drop in drops]
    lr_values = [lr] + [lr / (10 ** (idx + 1)) for idx, _ in enumerate(drops)]
    wd_values = [weight_decay * lr for lr in lr_values]

    lr_schedule = tf.optimizers.schedules.PiecewiseConstantDecay(boundaries, lr_values)

    wd_schedule = tf.optimizers.schedules.PiecewiseConstantDecay(boundaries, wd_values)

    return lr_schedule, wd_schedule


def init_training(args):
    """Initialize DETR training procedure

    Parameters
    ----------
    args : argparse.Namespace
        Arguments given to the program execution
    """
    if args.use_gpu:
        assert tf.config.list_physical_devices("GPU"), "No GPU available"
        assert tf.test.is_built_with_cuda(), "Tensorflow not compiled with CUDA support"

    # Get image input shape and number of images in path
    input_shape, count_images = get_image_information(args.storage_path)

    # Init Backbone Config
    backbone_config = {
        "input_shape": input_shape,
        "include_top": False,
        "weights": "imagenet",
    }

    # Init RPN Model
    detr = DETR(
        storage_path=args.storage_path,
        input_shape=input_shape,
        batch_size=args.batch_size,
        num_queries=args.num_queries,
        num_classes=args.num_classes,
        num_heads=args.num_heads,
        dim_transformer=args.dim_transformer,
        dim_feedforward=args.dim_feedforward,
        num_transformer_layer=args.num_transformer_layer,
        backbone_name=args.backbone_name,
        backbone_config=backbone_config,
        train_backbone=args.train_backbone,
    )

    num_steps = count_images // args.batch_size
    lr_schedule, wd_schedule = get_decay_schedules(
        num_steps, args.learning_rate, args.drops, args.weight_decay
    )

    optimizer = tfa.optimizers.AdamW(
        weight_decay=wd_schedule, learning_rate=lr_schedule
    )

    detr.train(
        epochs=args.epochs,
        optimizer=optimizer,
        batch_size=args.batch_size,
        count_images=count_images,
        use_pretrained=args.use_pretrained,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "DETR training script", parents=[get_args_parser()]
    )
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    init_training(args)
