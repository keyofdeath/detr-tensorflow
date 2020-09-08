"""Transformer MultiHeadAttention Class.

Taken and adjusted from: https://www.tensorflow.org/tutorials/text/transformer#multi-head_attention
"""

import tensorflow as tf
from detr_models.transformer.utils import scaled_dot_product_attention, split_heads


class MultiHeadAttention(tf.keras.Model):
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

    def call(self, inputs):
        v, k, q = inputs
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


class MultiHeadAttentionMap(tf.keras.Model):
    def __init__(self, num_heads: int, dim_transformer: int, dropout: float = 0.0):
        """Initialize MultiHeadAttentionMap.

        Parameters
        ----------
        num_heads : int
            Number of heads in multi-head attention layers.
        dim_transformer : int
            Number of neurons in multi-head attention layers.
            Should be a multiple of `num_heads`.
        dropout : float, optional
            Dropout probability to use.
        """
        super(MultiHeadAttentionMap, self).__init__()

        self.num_heads = num_heads
        self.dim_transformer = dim_transformer
        self.depth = dim_transformer // self.num_heads

        initializer = tf.keras.initializers.GlorotUniform()
        self.wq = tf.keras.layers.Dense(dim_transformer, kernel_initializer=initializer)
        self.wk = tf.keras.layers.Dense(dim_transformer, kernel_initializer=initializer)

        self.normalize_fact = float(dim_transformer / num_heads) ** -0.5
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, inputs):
        """Forward pass of MultiHeadAttentionMap.

        Parameters
        ----------
        k : tf.Tensor
            Key tensor of shape (Batch Size, H_4, W_4, d), where `H4` and `W_4` are the shapes
            of the last feature map and `d` the dimension of the transformer.
        q : tf.Tensor
            Query tensor of shape (Batch Size, #Queries, d), where `d` the dimension of the
            transformer.

        Returns
        -------
        attention_weights : tf.Tensor
            Attention heatmaps of shape (Batch Size, #Queries, #Heads, H_4, W_4).
            For each object, there are #Heads heatmaps in the shape of the last feature map.
        """
        k, q = inputs

        _, h, w, d = k.shape
        bs = tf.shape(k)[0]

        q = self.wq(q)
        k = self.wk(k)

        q = split_heads(q, self.num_heads, self.depth)
        k = split_heads(k, self.num_heads, self.depth)

        k = tf.reshape(k, shape=(bs, self.num_heads, h, w, self.depth))

        attention_weights = tf.einsum("bnqd,bnhwd->bqnhw", q * self.normalize_fact, k)

        attention_weights = tf.nn.softmax(attention_weights, axis=-1)
        attention_weights = self.dropout(attention_weights)

        return attention_weights
