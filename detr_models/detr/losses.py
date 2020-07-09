# IPDB can be used for debugging.
# Ignoring flake8 error code F401
import ipdb  # noqa: F401
import keras.backend as K
import tensorflow as tf
import tensorflow_addons as tfa
from detectors.detr.utils import box_cxcywh_to_xyxy

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
        Sample classification/score loss.
    """

    # Retrieve query and object idx
    query_idx, object_idx = filter_sample_indices(indices)

    # Filter target cls to aligne with matching outputs
    # Indices of Output
    out_indices = tf.expand_dims(query_idx, 1)

    # Updates of Target
    tgt_updates = tf.gather(target, object_idx)

    # Placeholder filled with only 'non_object'
    ordered_target_mask = tf.cast(
        tf.fill(dims=target.shape, value=4.0), dtype=tf.float32
    )

    # Ordered output according to hungarian matching
    # The target values of the labels are inserted at the idx corresponding
    # to the matching model output
    ordered_target = tf.tensor_scatter_nd_update(
        ordered_target_mask, out_indices, tgt_updates
    )

    sample_loss = K.sparse_categorical_crossentropy(
        target=ordered_target, output=output
    )

    # Balance between positive and negative predcitions, as there are way
    # more negative ones which would otherwise outweigh the result
    balancing_weights = tf.cast(tf.fill(dims=target.shape, value=0.1), dtype=tf.float32)
    balancing_weights = tf.tensor_scatter_nd_update(
        balancing_weights, out_indices, tf.ones_like(tgt_updates, dtype=tf.float32)
    )

    return tf.tensordot(sample_loss, balancing_weights, 1)


@tf.function
def bbox_loss(
    target,
    output,
    indices,
    bbox_cost_factor=tf.constant(5.0, dtype=tf.float32),
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
    bbox_cost_factor : tf.Tensor, optional
        Cost factor for L1-loss.
    iou_cost_factor : tf.Tensor, optional
        Cost factor for generalized IoU loss.

    Returns
    -------
    tf.Tensor
        Sample bounding box loss.
    """

    # Retrieve query and object idx
    query_idx, object_idx = filter_sample_indices(indices)

    # Select Ordered Target/Output according to the hungarian matching
    ordered_target = tf.gather(target, object_idx)
    ordered_output = tf.gather(output, query_idx)

    # Calculate L1 Loss
    l1_loss = tf.reduce_sum(tf.math.abs(ordered_target - ordered_output))

    # Calculate GIoU Loss
    giou_loss = tfa.losses.GIoULoss(reduction=tf.keras.losses.Reduction.NONE)
    giou_loss = tf.reduce_sum(
        giou_loss(
            y_true=box_cxcywh_to_xyxy(ordered_target),
            y_pred=box_cxcywh_to_xyxy(ordered_output),
        )
    )

    return bbox_cost_factor * l1_loss + iou_cost_factor * giou_loss
