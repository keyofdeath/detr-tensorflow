# IPDB can be used for debugging.
# Ignoring flake8 error code F401
import ipdb  # noqa: F401
import keras.backend as K
import tensorflow as tf
import tensorflow_addons as tfa
from detr_models.detr.utils import box_x1y1wh_to_yxyx

tf.keras.backend.set_floatx("float32")


@tf.function
def filter_sample_indices(indices):
    """Filter the correct indices that belong to the corresponding sample. The input tensor
    is padded to constitute a regular tensor. Therefore, the main goal of this function is
    to filter the matching query and object indices from the paddings.

    Parameters
    ----------
    indices : tf.Tensor
        Bipartite matching indices for the sample of shape [2, max_obj].
        Indicating the assignement between queries and objects. Note that `max_obj` is specified
        in `tf_linear_sum_assignment` and the tensor is padded with `-1`.

    Returns
    -------
    tf.Tensor, tf.Tensor
        Matching query and object indices filtered from the padded input tensor. Both tensors
        are of shape [#SampleObjects].
    """
    query_idx = tf.gather(indices, 0)
    object_idx = tf.gather(indices, 1)

    query_idx = tf.gather(query_idx, tf.where(tf.math.not_equal(query_idx, -1)))
    object_idx = tf.gather(object_idx, tf.where(tf.math.not_equal(object_idx, -1)))

    query_idx = tf.reshape(query_idx, shape=[tf.shape(query_idx)[0]])
    object_idx = tf.reshape(object_idx, shape=[tf.shape(object_idx)[0]])

    return query_idx, object_idx


@tf.function
def score_loss(target, output, indices):
    """Calculate classification/score loss.

    Parameters
    ----------
    target : tf.Tensor
        Sample target classes of shape [#Queries, 1].
    output : tf.Tensor
        Sample output class predictions of shape [#Queries, #Classes+1].
    indices : tf.Tensor
        Bipartite matching indices for the sample of shape [2, max_obj].
        Indicating the assignement between queries and objects. Note that `max_obj` is specified
        in `tf_linear_sum_assignment` and the tensor is padded with `-1`.

    Returns
    -------
    tf.Tensor
        Sample classification/score loss. Averaged over number of queries.
    """
    num_queries = tf.shape(output)[0]
    num_classes = tf.shape(output)[1]

    # Retrieve query and object idx
    query_idx, object_idx = filter_sample_indices(indices)

    # Filter target cls to aligne with matching outputs
    # Indices of Output
    out_indices = tf.expand_dims(query_idx, 1)

    # Updates of Target
    tgt_updates = tf.gather(target, object_idx)
    tgt_updates = tf.cast(tgt_updates, dtype=tf.int32)
    tgt_updates = tf.one_hot(tgt_updates, depth=num_classes, dtype=tf.float32)
    tgt_updates = tf.squeeze(tgt_updates, axis=1)

    # Placeholder filled with only 'non_object'
    # Balance between positive and negative predcitions, as there are way
    # more negative ones which would otherwise outweigh the result
    fill_tensor = tf.expand_dims(
        tf.one_hot(num_classes - 1, num_classes, on_value=0.1), 0
    )
    ordered_target_mask = tf.repeat(fill_tensor, repeats=num_queries, axis=0)

    # Ordered output according to hungarian matching
    # The target values of the labels are inserted at the idx corresponding
    # to the matching model output
    ordered_target = tf.tensor_scatter_nd_update(
        ordered_target_mask, out_indices, tgt_updates
    )

    sample_loss = K.categorical_crossentropy(target=ordered_target, output=output)
    return tf.reduce_mean(sample_loss)


@tf.function
def bbox_loss(
    target,
    output,
    indices,
    l1_cost_factor=tf.constant(5.0, dtype=tf.float32),
    iou_cost_factor=tf.constant(2.0, dtype=tf.float32),
):
    """Calculate bounding box loss.

    Parameters
    ----------
    target : tf.Tensor
        Sample target bounding boxes of shape [#Queries, 4].
    output : tf.Tensor
        Sample output bounding boxes predictions of shape [#Queries, 4].
    indices : tf.Tensor
        Bipartite matching indices for the sample of shape [2, max_obj].
        Indicating the assignement between queries and objects. Note that `max_obj` is specified
        in `tf_linear_sum_assignment` and the tensor is padded with `-1`.
    l1_cost_factor : tf.Tensor, optional
        Cost factor for L1-loss.
    iou_cost_factor : tf.Tensor, optional
        Cost factor for generalized IoU loss.

    Returns
    -------
    tf.Tensor
        Sample mean bounding box loss. Averaged over number of objects in sample.
    """
    # Retrieve query and object idx
    query_idx, object_idx = filter_sample_indices(indices)

    # Select Ordered Target/Output according to the hungarian matching
    ordered_target = tf.gather(target, object_idx)
    ordered_output = tf.gather(output, query_idx)
    # Calculate L1 Loss
    l1_loss = tf.reduce_sum(tf.math.abs(ordered_target - ordered_output), axis=1)

    # Calculate GIoU Loss
    giou_loss = tfa.losses.GIoULoss(reduction=tf.keras.losses.Reduction.NONE)

    giou_loss = 1 - tfa.losses.giou_loss(
        y_true=box_x1y1wh_to_yxyx(ordered_target),
        y_pred=box_x1y1wh_to_yxyx(ordered_output),
    )

    return tf.reduce_mean(l1_cost_factor * l1_loss + iou_cost_factor * giou_loss)
