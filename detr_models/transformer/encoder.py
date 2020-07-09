"""Transformer Encoder Class.

Taken and adjusted from: https://www.tensorflow.org/tutorials/text/transformer#multi-head_attention
"""

import ipdb  # noqa: F401
import tensorflow as tf
from detectors.transformer.attention import MultiHeadAttention


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(
        self, num_layers, dim_transformer, num_heads, dim_feedforward, dropout=0.1
    ):
        super(TransformerEncoder, self).__init__()

        self.dim_transformer = dim_transformer
        self.num_layers = num_layers

        self.enc_layers = [
            TransformerEncoderLayer(
                dim_transformer, num_heads, dim_feedforward, dropout
            )
            for _ in range(num_layers)
        ]

    def __call__(self, src, positional_encodings, training):

        enc_output = src

        for i in range(self.num_layers):
            enc_output = self.enc_layers[i](enc_output, positional_encodings, training)

        return enc_output


class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, dim_transformer, num_heads, dim_feedforward, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()

        self.selt_attn = MultiHeadAttention(dim_transformer, num_heads)

        self.linear1 = tf.keras.layers.Dense(dim_feedforward, activation="relu")
        self.linear2 = tf.keras.layers.Dense(dim_transformer)

        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

    def __call__(self, src, positional_encodings, training=True):

        q = k = src + positional_encodings

        attn_output, attention_weights = self.selt_attn(src, k, q)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.norm1(src + attn_output)

        ffn_output = self.linear2(self.dropout2(self.linear1(out1)))
        ffn_output = self.dropout2(ffn_output, training=training)

        out1 = tf.cast(out1, dtype=tf.float32)
        ffn_output = tf.cast(ffn_output, dtype=tf.float32)

        out2 = self.norm2(out1 + ffn_output)

        return out2
