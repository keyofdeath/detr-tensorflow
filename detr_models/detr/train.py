# IPDB can be used for debugging.
# Ignoring flake8 error code F401
import ipdb  # noqa: F401
import tensorflow as tf
from detectors.detr.config import DETRTrainingConfig
from detectors.detr.model import DETR

tf.keras.backend.set_floatx("float32")

# os.environ['AUTOGRAPH_VERBOSITY'] = "1"


config = DETRTrainingConfig()


if config.run_on_gpu:
    assert tf.config.list_physical_devices("GPU"), "No GPU available"
    assert tf.test.is_built_with_cuda(), "Tensorflow not compiled with CUDA support"

# Init RPN Model
detr = DETR(
    storage_path=config.storage_path,
    input_shape=config.input_shape,
    num_queries=config.num_queries,
    num_classes=config.num_classes,
    num_heads=config.num_heads,
    dim_transformer=config.dim_transformer,
    dim_feedforward=config.dim_feedforward,
    num_transformer_layer=config.num_transformer_layer,
    backbone_name=config.backbone_name,
    backbone_config=config.backbone_config,
    train_backbone=False,
)

optimizer = tf.keras.optimizers.Adam(config.learning_rate)
detr.train(
    epochs=config.epochs,
    optimizer=optimizer,
    batch_size=config.batch_size,
    count_images=config.count_images,
)
