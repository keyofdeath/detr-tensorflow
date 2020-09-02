import os

# IPDB can be used for debugging.
# Ignoring flake8 error code F401
import ipdb  # noqa: F401
import tensorflow as tf
import tensorflow_addons as tfa
from detr_models.data_feeder.coco_feeder import COCOFeeder
from detr_models.data_feeder.pvoc_feeder import PVOCFeeder
from detr_models.detr.config import DefaultDETRConfig
from detr_models.detr.model import DETR
from tensorflow.keras.preprocessing.image import img_to_array, load_img

tf.keras.backend.set_floatx("float32")

model_config = DefaultDETRConfig()


def get_image_information(storage_path: str, data_type: str):
    """Helper function to retrieve image information.

    Parameters
    ----------
    storage_path : str
        Path to data storage.
    data_type : str
        Model configuration specifying data_type (`PVOC` or `COCO`).

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

    if data_type == "PVOC":
        input_shape = sample_image.shape
    elif data_type == "COCO":
        input_shape = (model_config.image_height, model_config.image_width, 3)

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

    if model_config.data_type not in ["PVOC", "COCO"]:
        raise ValueError(
            "Invalid `data_type` specified in Config: {}".format(model_config.data_type)
        )

    # Get image input shape and number of images in path
    input_shape, count_images = get_image_information(
        training_config["storage_path"], model_config.data_type
    )

    # Init Backbone Config
    backbone_config = {
        "input_shape": input_shape,
        "include_top": False,
        "weights": "imagenet",
    }

    # Init RPN Model
    detr = DETR(
        input_shape=input_shape,
        num_queries=model_config.num_queries,
        num_classes=model_config.num_classes,
        num_heads=model_config.num_heads,
        dim_transformer=model_config.dim_transformer,
        dim_feedforward=model_config.dim_feedforward,
        num_transformer_layer=model_config.num_transformer_layer,
        train_backbone=model_config.train_backbone,
        backbone_name=model_config.backbone_name,
        backbone_config=backbone_config,
    )

    if model_config.data_type == "PVOC":
        print("Use Input Data in PascalVOC format")
        data_feeder = PVOCFeeder(
            storage_path=training_config["storage_path"],
            batch_size=training_config["batch_size"],
            num_queries=model_config.num_queries,
            num_classes=model_config.num_classes,
        )
    else:
        print("Use Input Data in COCO format")
        data_feeder = COCOFeeder(
            storage_path=training_config["storage_path"],
            batch_size=training_config["batch_size"],
            num_queries=model_config.num_queries,
            num_classes=model_config.num_classes,
            image_width=model_config.image_width,
            image_height=model_config.image_height,
        )

    lr_schedule, wd_schedule = get_decay_schedules(
        num_steps=count_images // training_config["batch_size"],
        lr=model_config.learning_rate,
        drops=model_config.drops,
        weight_decay=model_config.weight_decay,
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
