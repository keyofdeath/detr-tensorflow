# IPDB can be used for debugging.
# Ignoring flake8 error code F401
import ipdb  # noqa: F401
import tensorflow as tf
from detr_models.detr.matcher import bipartite_matching
from detr_models.detr.losses import (
    calculate_cls_loss,
    calculate_bbox_loss,
    calculate_mask_loss,
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

        cls_loss = calculate_cls_loss(batch_cls, detr_scores, indices)

        l1_loss, giou_loss = calculate_bbox_loss(batch_bbox, detr_bbox, indices)
        bbox_loss = tf.constant(5.0) * l1_loss + tf.constant(2.0) * giou_loss

        detr_loss = cls_loss + bbox_loss

        gradients = gradient_tape.gradient(detr_loss, detr.trainable_variables)
        gradients = [
            tf.clip_by_norm(gradient, tf.constant(0.1)) for gradient in gradients
        ]
        optimizer.apply_gradients(zip(gradients, detr.trainable_variables))

    mask_loss = tf.constant(0.0)

    return [detr_loss, cls_loss, bbox_loss, mask_loss]


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
    batch_masks : tf.Tensor
        Batch segmentation masks of shape [Batch size, #Objects, H, W]
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

        cls_loss = calculate_cls_loss(batch_cls, detr_scores, indices)
        ipdb.set_trace()
        l1_loss, giou_loss = calculate_bbox_loss(batch_bbox, detr_bbox, indices)
        bbox_loss = tf.constant(5.0) * l1_loss + tf.constant(2.0) * giou_loss

        dice_loss, focal_loss = calculate_mask_loss(batch_masks, detr_masks, indices)
        mask_loss = dice_loss + focal_loss

        detr_loss = cls_loss + bbox_loss + mask_loss

        gradients = gradient_tape.gradient(detr_loss, detr.trainable_variables)
        gradients = [
            tf.clip_by_norm(gradient, tf.constant(0.1)) for gradient in gradients
        ]
        optimizer.apply_gradients(zip(gradients, detr.trainable_variables))

    return [detr_loss, cls_loss, bbox_loss, mask_loss]
