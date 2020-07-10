"""Transformer Class.

Mostly taken from: https://www.tensorflow.org/tutorials/text/transformer#multi-head_attention
"""

import tensorflow as tf
from detr_models.transformer.decoder import TransformerDecoder
from detr_models.transformer.encoder import TransformerEncoder

tf.keras.backend.set_floatx("float32")


class Transformer(tf.keras.Model):
    def __init__(
        self, num_layers, dim_transformer, num_heads, dim_feedforward, dropout=0.1
    ):
        super(Transformer, self).__init__()

        self.num_layers = num_layers
        self.dim_transformer = dim_transformer
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward

        self.encoder = TransformerEncoder(
            num_layers, dim_transformer, num_heads, dim_feedforward, dropout
        )
        self.decoder = TransformerDecoder(
            num_layers, dim_transformer, num_heads, dim_feedforward, dropout
        )

    def __call__(self, inp, positional_encodings, query_pos, training=True):

        tgt = tf.zeros_like(query_pos)

        memory = self.encoder(inp, positional_encodings, training)

        hidden_space = self.decoder(
            tgt, memory, positional_encodings, query_pos, training
        )

        return hidden_space
