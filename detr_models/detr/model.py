import time
import warnings

# IPDB can be used for debugging.
# Ignoring flake8 error code F401
import ipdb  # noqa: F401
import numpy as np
import tensorflow as tf

from detr_models.backbone.backbone import Backbone
from detr_models.transformer.transformer import Transformer
from detr_models.detr.utils import create_positional_encodings, save_training_loss
from detr_models.detr.train import train_base_model, train_segmentation_model

from detr_models.transformer.attention import MultiHeadAttentionMap
from detr_models.detr.segmentation import SegmentationHead

warnings.filterwarnings("ignore")

tf.keras.backend.set_floatx("float32")


class DETR:
    def __init__(
        self,
        input_shape,
        num_queries,
        num_classes,
        num_heads,
        dim_transformer,
        dim_feedforward,
        num_transformer_layer,
        backbone_name,
        backbone_config,
        train_backbone,
    ):
        """Initialize Detection Transformer (DETR) network.

        Parameters
        ----------
        input_shape : tuple
            Specification of model input [H, W, C].
        num_queries : int
            Number of queries used in transformer.
        num_classes : int
            Number of target classes.
        num_heads : int
            Number of heads in multi-head attention layers.
        dim_transformer : int
            Number of neurons in multi-head attention layers.
            Should be a multiple of `num_heads`.
        dim_feedforward : int
            Number of neurons in transformer feed forward layers.
        num_transformer_layer : int
            Number of layers in transformer network.
        backbone_name : str
            Name of backbone used for DETR network.
        backbone_config : dict
            Config of backbone used for DETR network.
        train_backbone : bool
            Flag to indicate training of backbone.
        """

        # Save object parameters
        self.input_shape = input_shape
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.dim_transformer = dim_transformer
        self.dim_feedforward = dim_feedforward
        self.num_transformer_layer = num_transformer_layer
        self.train_backbone = train_backbone

        # Init Backbone
        self.backbone = Backbone(backbone_name, backbone_config, True).model
        self.backbone.trainable = train_backbone
        self.fm_shape = self.backbone.get_layer("feature_map").output.shape[1::]
        self.positional_encodings_shape = (
            self.fm_shape[0] * self.fm_shape[1],
            dim_transformer,
        )

        # Init Layers
        self.embedding_layer = tf.keras.layers.Embedding(
            input_dim=self.num_queries, output_dim=self.dim_transformer
        )

        self.fm_projection = tf.keras.layers.Conv2D(
            dim_transformer, kernel_size=1, name="FeatureMap_Projection"
        )
        self.transformer = Transformer(
            num_transformer_layer, dim_transformer, num_heads, dim_feedforward
        )

        self.cls_head = tf.keras.layers.Dense(
            units=self.num_classes + 1, activation="softmax", name="cls_head"
        )

        self.bbox_head = tf.keras.layers.Dense(
            units=4, activation="sigmoid", name="bbox_head"
        )

        self.attention_layer = MultiHeadAttentionMap(
            dim_transformer=dim_transformer, num_heads=num_heads
        )

        self.segmentation_head = SegmentationHead(
            num_heads=num_heads, dim_transformer=dim_transformer
        )

    def build_model(self, use_pretrained=False, train_masks=False):
        """Build Detection Transformer (DETR) model."""

        if use_pretrained:
            print("\nLoad pre-trained model: {}".format(use_pretrained))
            self.model = tf.keras.models.load_model(use_pretrained, compile=False)
        else:
            print("\nBuild Model from scratch.")
            batch_input = tf.keras.layers.Input(
                shape=self.input_shape, name="Batch_Input"
            )
            positional_encodings = tf.keras.layers.Input(
                shape=self.positional_encodings_shape, name="Positional_Encodings_Input"
            )

            fpn_maps = self.backbone(batch_input)
            feature_map = fpn_maps[-1]

            # Set backbone learning rate order of magnitude smaller
            feature_map = (1 / 10) * feature_map + (1 - 1 / 10) * tf.stop_gradient(
                feature_map
            )

            transformer_input = self.fm_projection(feature_map)

            batch_size = tf.shape(transformer_input)[0]

            # Create Queries
            # Query Input is always a tensor of ones, therefore the output
            # equals the weights of the Embedding Layer
            query_pos = tf.ones((self.num_queries), dtype=tf.float32)
            query_pos = tf.repeat(
                tf.expand_dims(query_pos, axis=0), repeats=batch_size, axis=0
            )
            query_embedding = self.embedding_layer(query_pos)

            transformer_output, memory = self.transformer(
                [transformer_input, positional_encodings, query_embedding]
            )

            cls_pred = self.cls_head(transformer_output)

            bbox_pred = self.bbox_head(transformer_output)

            output_tensor = [cls_pred, bbox_pred]

            if train_masks:
                attention_hmaps = self.attention_layer([memory, transformer_output])
                mask_pred = self.segmentation_head(
                    [transformer_input, attention_hmaps, fpn_maps[:-1]]
                )

                output_tensor = [cls_pred, bbox_pred, mask_pred]

            self.model = tf.keras.Model(
                [batch_input, positional_encodings], output_tensor, name="DETR"
            )

    def train(self, training_config, optimizer, count_images, data_feeder):
        """Train the DETR Model.

        Parameters
        ----------
        training_config : dict
            Contains the training configuration:
        optimizer : tf.Optimizer
            Any chosen optimizer used for training.
        count_images : int
            Number of total images used for training.
        data_feeder : detr_models.detr.data_feeder
            DataFeeder object used for training. Currently, we supprt a data feeder taking
            input data of PascalVOC and COCO type.

        Returns
        -------
        float
            Final training loss.
        """
        print(
            "\nStart Training - Total of {} Epochs:\n".format(
                training_config["epochs"]
            ),
            flush=True,
        )

        detr_loss = []

        positional_encodings = create_positional_encodings(
            fm_shape=self.fm_shape,
            num_pos_feats=self.dim_transformer // 2,
            batch_size=training_config["batch_size"],
        )

        if training_config["train_masks"] and (len(self.model.output) != 3):
            raise TypeError(
                "Segmentation Head is not available in model. Please build model with head to train masks."
            )

        for epoch in range(training_config["epochs"]):
            start = time.time()
            print("-------------------------------------------", flush=True)
            print(f"Epoch: {epoch+1}\n", flush=True)
            epoch_loss = np.array([0.0, 0.0, 0.0])
            batch_iteration = 0

            # Iterate over all batches
            for input_data in data_feeder(training_config["verbose"]):

                if training_config["train_masks"]:
                    batch_loss = train_segmentation_model(
                        detr=self.model,
                        optimizer=optimizer,
                        batch_inputs=input_data[0],
                        batch_cls=input_data[1],
                        batch_bbox=input_data[2],
                        batch_masks=input_data[4],
                        obj_indices=input_data[3],
                        positional_encodings=positional_encodings,
                    )
                else:
                    batch_loss = train_base_model(
                        detr=self.model,
                        optimizer=optimizer,
                        batch_inputs=input_data[0],
                        batch_cls=input_data[1],
                        batch_bbox=input_data[2],
                        obj_indices=input_data[3],
                        positional_encodings=positional_encodings,
                    )

                batch_loss = np.array([loss.numpy() for loss in batch_loss])

                if training_config["verbose"]:
                    print("\nDETR Batch Loss: {:.3f}".format(batch_loss[0]))
                    print("Cls. Batch Loss: {:.3f}".format(batch_loss[1]))
                    print("Bounding Box Batch Loss: {:.3f}\n".format(batch_loss[2]))

                epoch_loss = epoch_loss + batch_loss

                batch_iteration += 1

            epoch_loss = (1 / batch_iteration) * epoch_loss
            detr_loss.append(epoch_loss)

            print("DETR Loss: {:.3f}".format(epoch_loss[0]), flush=True)
            print("Class Loss: {:.3f}".format(epoch_loss[1]), flush=True)
            print("Bounding Box Loss: {:.3f}".format(epoch_loss[2]), flush=True)

            print(f"Time for epoch {epoch + 1} is {time.time()-start} sec", flush=True)
            print("-------------------------------------------\n", flush=True)

        print("Finalize Training\n", flush=True)

        # Save training loss and model
        self.model.save("{}/detr_model".format(training_config["output_dir"]))
        save_training_loss(
            detr_loss, "{}/detr_loss.txt".format(training_config["output_dir"])
        )

        return detr_loss
