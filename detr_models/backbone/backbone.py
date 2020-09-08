import tensorflow as tf
from tensorflow.keras import Model
from voluptuous import Required, Schema, Optional

tf.keras.backend.set_floatx("float32")


class Backbone:
    def __init__(self, backbone_name, config, return_intermediate=False):
        """Initialize Backbone Model. You can choose between MobileNetV2,ResNet50
        and InceptionV3.

        Parameters
        ----------
        backbone_name : str
            Name of backbone model to use. Either MobileNetV2,ResNet50 or InceptionV3.
        config : dict
            Ditionary to config backbone model. Has to follow the specified schema definition.
        """

        backbone_config = Schema(
            {
                Required("input_shape"): Schema((int, int, int)),
                Required("include_top"): bool,
                Required("weights"): str,
                Optional("alpha"): float,
            }
        )

        config = backbone_config(config)

        if backbone_name == "MobileNetV2":
            self.model = tf.keras.applications.MobileNetV2(**config)
        elif backbone_name == "ResNet50":
            self.model = tf.keras.applications.ResNet50(**config)
        elif backbone_name == "InceptionV3":
            self.model = tf.keras.applications.InceptionV3(**config)

        # Remove Layers until Conv4
        for i, layer in enumerate(reversed(self.model.layers)):
            if backbone_name == "ResNet50" and layer._name == "conv4_block6_out":
                break
            elif (
                backbone_name == "MobileNetV2" and layer._name == "block_13_expand_relu"
            ):
                break
            else:
                self.model._layers.pop()

        self.model.layers[-1]._name = "feature_map"

        output = get_output_layers(self.model.layers, return_intermediate)
        self.model = Model(self.model.input, output, name="Backbone")


def get_output_layers(layers: list, return_intermediate: bool):
    """Retrieve the backbone output to return for DETR.

    Parameters
    ----------
    layers : list
        Backbone layers.
    return_intermediate : bool
        Flag indicating whether intermediate layers should be returned.

    Returns
    -------
    list
        List of tf.Layers to be returned from backbone.
    """
    if return_intermediate:
        layer_names = [
            "conv1_relu",
            "conv2_block3_out",
            "conv3_block4_out",
            "feature_map",
        ]
    else:
        layer_names = ["feature_map"]

    layer_idxs = [idx for idx, layer in enumerate(layers) if layer.name in layer_names]
    return [layers[idx].output for idx in layer_idxs]
