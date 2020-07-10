"""Transformer Decoder Class.

Taken and adjusted from: https://www.tensorflow.org/tutorials/text/transformer#multi-head_attention
"""

import ipdb  # noqa: F401
import tensorflow as tf
from detr_models.transformer.attention import MultiHeadAttention


class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(
        self, num_layers, dim_transformer, num_heads, dim_feedforward, dropout=0.1
    ):
        super(TransformerDecoder, self).__init__()

        self.dim_transformer = dim_transformer
        self.num_layers = num_layers

        self.dec_layers = [
            TransformerDecoderLayer(
                dim_transformer, num_heads, dim_feedforward, dropout
            )
            for _ in range(num_layers)
        ]

    def __call__(self, tgt, memory, positional_encodings, query_pos, training):

        dec_output = tgt

        for i in range(self.num_layers):
            dec_output = self.dec_layers[i](
                dec_output, memory, positional_encodings, query_pos, training
            )

        return dec_output


class TransformerDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, dim_transformer, num_heads, dim_feedforward, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(dim_transformer, num_heads)
        self.multihead_attn = MultiHeadAttention(dim_transformer, num_heads)

        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    dim_feedforward, activation="relu"
                ),  # (batch_size, seq_len, dim_feedforward)
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(
                    dim_transformer
                ),  # (batch_size, seq_len, dim_transformer)
            ]
        )

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        self.dropout3 = tf.keras.layers.Dropout(dropout)

    def __call__(self, tgt, memory, positional_encodings, query_pos, training=True):
        # memory.shape == (batch_size, input_seq_len, dim_transformer)
        q = k = tf.math.add(tgt, query_pos, "Decoder_Add_1")

        # Selt Attention
        attn1, attn_weights_block1 = self.self_attn(v=tgt, k=k, q=q)

        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(tf.math.add(attn1, tgt, "Decoder_Add_2"))

        # Multi-Head Attention
        attn2, attn_weights_block2 = self.multihead_attn(
            v=memory,
            k=tf.math.add(memory, positional_encodings, "Decoder_Add_3"),
            q=tf.math.add(out1, query_pos, "Decoder_Add_4"),
        )
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(tf.math.add(attn2, out1, "Decoder_Add_5"))

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)

        out3 = self.layernorm3(tf.math.add(ffn_output, out2, "Decoder_Add_6"))

        return out3
