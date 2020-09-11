# IPDB can be used for debugging.
# Ignoring flake8 error code F401
import ipdb  # noqa: F401
import tensorflow as tf
from detr_models.detr.matcher import bipartite_matching
from detr_models.detr.losses import (
    bbox_loss,
    score_loss,
    calculate_dice_loss,
    calculate_focal_loss,
)


@tf.function
def train_base_model(
    detr,
    optimizer,
    batch_inputs,
    batch_cls,
    batch_bbox,
    obj_indices,
    positional_encodings,
):
    """Train step of the DETR network.

    Parameters
    ----------
    detr : tf.Model
        Detection Transformer (DETR) Model without segmentation head.
    optimizer : tf.Optimizer
        Any chosen optimizer used for training.
    batch_inputs : tf.Tensor
        Batch input images of shape [Batch Size, H, W, C].
    batch_cls : tf.Tensor
        Batch class targets of shape [Batch Size, #Queries, 1].
    batch_bbox : tf.Tensor
        Batch bounding box targets of shape [Batch Size, #Queries, 4].
    obj_indices : tf.RaggedTensor
        Helper tensor of shape [Batch Size, None].
        Used to link objects in the cost matrix to the target tensors.
    positional_encodings : tf.Tensor
        Positional encodings of shape [Batch Size, H*W, dim_transformer].
        Used in transformer network to enrich input information.
    """

    with tf.GradientTape() as gradient_tape:
        detr_scores, detr_bbox = detr(
            [batch_inputs, positional_encodings], training=True
        )

        indices = bipartite_matching(
            detr_scores, detr_bbox, batch_cls, batch_bbox, obj_indices
        )

        score_loss = tf.constant(10.0) * calculate_score_loss(
            batch_cls, detr_scores, indices
        )
        bbox_loss = calculate_bbox_loss(batch_bbox, detr_bbox, indices)

        detr_loss = score_loss + bbox_loss

        gradients = gradient_tape.gradient(detr_loss, detr.trainable_variables)
        gradients = [
            tf.clip_by_norm(gradient, tf.constant(0.1)) for gradient in gradients
        ]
        optimizer.apply_gradients(zip(gradients, detr.trainable_variables))

    return [detr_loss, score_loss, bbox_loss]


@tf.function
def train_segmentation_model(
    detr,
    optimizer,
    batch_inputs,
    batch_cls,
    batch_bbox,
    batch_masks,
    obj_indices,
    positional_encodings,
):
    """Train step of the DETR network.

    Parameters
    ----------
    detr : tf.Model
        Detection Transformer (DETR) Model including the segmentation head.
    optimizer : tf.Optimizer
        Any chosen optimizer used for training.
    batch_inputs : tf.Tensor
        Batch input images of shape [Batch Size, H, W, C].
    batch_cls : tf.Tensor
        Batch class targets of shape [Batch Size, #Queries, 1].
    batch_bbox : tf.Tensor
        Batch bounding box targets of shape [Batch Size, #Queries, 4].
    obj_indices : tf.RaggedTensor
        Helper tensor of shape [Batch Size, None].
        Used to link objects in the cost matrix to the target tensors.
    positional_encodings : tf.Tensor
        Positional encodings of shape [Batch Size, H*W, dim_transformer].
        Used in transformer network to enrich input information.
    """

    with tf.GradientTape() as gradient_tape:
        detr_scores, detr_bbox, detr_masks = detr(
            [batch_inputs, positional_encodings], training=True
        )

        indices = bipartite_matching(
            detr_scores, detr_bbox, batch_cls, batch_bbox, obj_indices
        )

        score_loss = tf.constant(10.0) * calculate_score_loss(
            batch_cls, detr_scores, indices
        )
        bbox_loss = calculate_bbox_loss(batch_bbox, detr_bbox, indices)

        mask_loss = calculate_mask_loss(batch_masks, detr_masks, indices)

        detr_loss = score_loss + bbox_loss + mask_loss

        gradients = gradient_tape.gradient(detr_loss, detr.trainable_variables)
        gradients = [
            tf.clip_by_norm(gradient, tf.constant(0.1)) for gradient in gradients
        ]
        optimizer.apply_gradients(zip(gradients, detr.trainable_variables))

    return [detr_loss, score_loss, bbox_loss]


def calculate_score_loss(batch_cls, detr_scores, indices):
    """Helper function to calculate the score loss.

    Parameters
    ----------
    batch_cls : tf.Tensor
        Batch class targets of shape [Batch Size, #Queries, 1].
    detr_scores : tf.Tensor
        Batch detr score outputs of shape [Batch Size. #Queries, #Classes + 1].
    indices : tf.Tensor
        Bipartite matching indices of shape [Batch Size, 2, max_obj].
        Indicating the assignement between queries and objects in each sample. Note that `max_obj` is
        specified in `tf_linear_sum_assignment` and the tensor is padded with `-1`.

    Returns
    -------
    tf.Tensor
        Average batch score loss.
    """
    batch_score_loss = tf.map_fn(
        lambda el: score_loss(*el),
        elems=[batch_cls, detr_scores, indices],
        dtype=tf.float32,
    )
    return tf.reduce_mean(batch_score_loss)


def calculate_bbox_loss(batch_bbox, detr_bbox, indices):
    """Helper function to calculate the bounding box loss.

    Parameters
    ----------
    batch_bbox : tf.Tensor
        Batch bounding box targets of shape [Batch Size, #Queries, 4].
    detr_bbox : tf.Tensor
        Batch detr bounding box outputs of shape [Batch Size. #Queries, 4].
    indices : tf.Tensor
        Bipartite matching indices of shape [Batch Size, 2, max_obj].
        Indicating the assignement between queries and objects in each sample. Note that `max_obj` is
        specified in `tf_linear_sum_assignment` and the tensor is padded with `-1`.

    Returns
    -------
    tf.Tensor
        Average batch bounding box loss
    """

    batch_bbox_loss = tf.map_fn(
        lambda el: bbox_loss(*el),
        elems=[batch_bbox, detr_bbox, indices],
        dtype=tf.float32,
    )
    return tf.reduce_mean(batch_bbox_loss)


def calculate_mask_loss(batch_masks, detr_masks, indices):
    """Calculate batch segmentation mask loss.

    Parameters
    ----------
    batch_masks : tf.Tensor
        Batch mask targets of shape [Batch Size, #Objects, H, W].
    detr_masks : tf.Tensor
        Batch detr mask outputs of shape [Batch Size, #Queries, H_1, W_1], where
            `H_1` and `W_1` are the shapes of the first feature map coming from
            the ResNet50 network.
    indices : tf.Tensor
        Bipartite matching indices of shape [Batch Size, 2, max_obj].
        Indicating the assignement between queries and objects in each sample. Note that `max_obj` is
        specified in `tf_linear_sum_assignment` and the tensor is padded with `-1`.

    Returns
    -------
    tf.Tensor
        Average batch mask loss.
    """
    query_idx = tf.where(tf.math.not_equal(indices[:, 0], -1))
    output = tf.gather_nd(detr_masks, query_idx)

    object_idx = tf.where(tf.math.not_equal(indices[:, 1], -1))
    target = tf.gather_nd(batch_masks, object_idx).to_tensor()

    output = tf.expand_dims(output, axis=-1)
    output = tf.image.resize(output, size=tf.shape(target)[-2::], method="nearest")
    output = tf.squeeze(output)

    target = tf.reshape(target, shape=(tf.shape(target)[0], -1))
    output = tf.reshape(output, shape=(tf.shape(output)[0], -1))

    dice_loss = calculate_dice_loss(target, output)
    focal_loss = calculate_focal_loss(target, output)

    mask_loss = dice_loss + focal_loss
    return mask_loss
