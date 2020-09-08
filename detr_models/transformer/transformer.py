"""Transformer Class.

Mostly taken from: https://www.tensorflow.org/tutorials/text/transformer#multi-head_attention
"""

import tensorflow as tf
import ipdb  # noqa: F401

from detr_models.transformer.decoder import TransformerDecoder
from detr_models.transformer.encoder import TransformerEncoder

tf.keras.backend.set_floatx("float32")


class Transformer(tf.keras.Model):
    def __init__(
        self,
        num_layers,
        dim_transformer,
        num_heads,
        dim_feedforward,
        dropout=0.1,
        name="Transformer",
    ):
        super(Transformer, self).__init__(name=name)

        self.num_layers = num_layers
        self.dim_transformer = dim_transformer
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward

        self.encoder = TransformerEncoder(
            num_layers,
            dim_transformer,
            num_heads,
            dim_feedforward,
            dropout,
            name="Encoder",
        )
        self.decoder = TransformerDecoder(
            num_layers,
            dim_transformer,
            num_heads,
            dim_feedforward,
            dropout,
            name="Decoder",
        )

    def call(self, inputs):
        inp, positional_encodings, query_pos = inputs

        _, h, w, d = inp.shape
        bs = tf.shape(inp)[0]

        # BS x H x W x C to BS x HW x C
        inp = tf.reshape(inp, shape=(bs, h * w, d))

        tgt = tf.zeros_like(query_pos)

        memory = self.encoder([inp, positional_encodings])

        hidden_space = self.decoder([tgt, memory, positional_encodings, query_pos])

        memory = tf.reshape(memory, shape=(bs, h, w, d))
        return hidden_space, memory
