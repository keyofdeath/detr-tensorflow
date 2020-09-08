"""Transformer utils.

Taken and adjusted from: https://www.tensorflow.org/tutorials/text/transformer#multi-head_attention
"""

import tensorflow as tf


def scaled_dot_product_attention(q, k, v):
    """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.

  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)

  Returns:
    output, attention_weights
  """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


def split_heads(x: tf.Tensor, num_heads: int, depth: int):
    """Split the last dimension into (num_heads, depth) and transpose the result

    Parameters
    ----------
    x : tf.Tensor
        Description
    batch_size : int
        Description
    num_heads : int
        Description
    depth : int
        Description

    Returns
    -------
    tf.Tensor
        Splitted encoding of shape (Batch Size, #Heads, X, depth), where `X` is the length
        of the sequence.
    """
    batch_size = tf.shape(x)[0]
    x = tf.reshape(x, (batch_size, -1, num_heads, depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])
