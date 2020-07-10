"""Transformer MultiHeadAttention Class.

Taken and adjusted from: https://www.tensorflow.org/tutorials/text/transformer#multi-head_attention
"""

import tensorflow as tf
from detr_models.transformer.utils import scaled_dot_product_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, dim_transformer, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.dim_transformer = dim_transformer
        self.depth = dim_transformer // self.num_heads

        self.wq = tf.keras.layers.Dense(dim_transformer)
        self.wk = tf.keras.layers.Dense(dim_transformer)
        self.wv = tf.keras.layers.Dense(dim_transformer)

        self.dense = tf.keras.layers.Dense(dim_transformer)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
            Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def __call__(self, v, k, q):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, dim_transformer)
        k = self.wk(k)  # (batch_size, seq_len, dim_transformer)
        v = self.wv(v)  # (batch_size, seq_len, dim_transformer)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v)

        scaled_attention = tf.transpose(
            scaled_attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(
            scaled_attention, (batch_size, -1, self.dim_transformer)
        )  # (batch_size, seq_len_q, dim_transformer)

        output = self.dense(
            concat_attention
        )  # (batch_size, seq_len_q, dim_transformer)

        return output, attention_weights
