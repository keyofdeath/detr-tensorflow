# IPDB can be used for debugging.
# Ignoring flake8 error code F401
import ipdb  # noqa: F401
import keras.backend as K
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.losses as tfl

from detr_models.detr.utils import box_x1y1wh_to_yxyx, extract_ordering_idx

tf.keras.backend.set_floatx("float32")


def calculate_cls_loss(
    batch_cls, detr_scores, indices, non_object_weight=tf.constant(0.1)
):
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
    num_classes = tf.math.reduce_max(batch_cls)
    query_idx = extract_ordering_idx(indices[:, 0])

    object_idx = extract_ordering_idx(indices[:, 1])
    target_updates = tf.gather_nd(batch_cls, object_idx)

    target = tf.fill(tf.shape(batch_cls), value=num_classes)
    target = tf.tensor_scatter_nd_update(target, query_idx, target_updates)

    object_weight = tf.fill(tf.shape(target_updates), value=tf.constant(1.0))
    sample_weight = tf.fill(tf.shape(batch_cls), value=non_object_weight)
    sample_weight = tf.tensor_scatter_nd_update(sample_weight, query_idx, object_weight)

    cross_entropy = calculate_ce_loss(target, detr_scores, sample_weight)
    return cross_entropy


def calculate_bbox_loss(batch_bbox, detr_bbox, indices):
    """Calculate batch bounding box losses.

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
    l1_loss : tf.Tensor[float]
        L1 loss averaged by the number of objects in batch.
    giou_loss : tf.Tensor[float]
        GIoU loss averaged by the number of objects in batch.
    """
    query_idx = extract_ordering_idx(indices[:, 0])
    output = tf.gather_nd(detr_bbox, query_idx)

    object_idx = extract_ordering_idx(indices[:, 1])
    target = tf.gather_nd(batch_bbox, object_idx)

    l1_loss = calculate_l1_loss(target, output)
    giou_loss = calculate_giou_loss(target, output)

    return l1_loss, giou_loss


def calculate_mask_loss(batch_masks, detr_masks, indices):
    """Calculate batch segmentation mask losses.

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
    query_idx = extract_ordering_idx(indices[:, 0])
    output = tf.gather_nd(detr_masks, query_idx)

    object_idx = extract_ordering_idx(indices[:, 1])
    target = tf.gather_nd(batch_masks, object_idx).to_tensor()

    output = tf.expand_dims(output, axis=-1)
    output = tf.image.resize(output, size=tf.shape(target)[-2::], method="nearest")
    output = tf.squeeze(output)

    target = tf.reshape(target, shape=(tf.shape(target)[0], -1))
    output = tf.reshape(output, shape=(tf.shape(output)[0], -1))

    dice_loss = calculate_dice_loss(target, output)
    focal_loss = calculate_focal_loss(target, output)

    return dice_loss, focal_loss


def calculate_ce_loss(target: tf.Tensor, output: tf.Tensor, sample_weight=None):
    """Calculate weighted Cross-Entropy loss for batch class predictions.

    Parameters
    ----------
    target : tf.Tensor
        Ordered target class labels of shape [Batch Size, #Queries, 1]
    output : tf.Tensor
        Output class predictions of shape [Batch Size, #Queries, #Classes + 1]
    sample_weight : None, optional
        Sample weight tensor of shape [Batch Size, #Queries].
        If provided, scales the loss of each query by the corresponding weight entry.

    Returns
    -------
    tf.Tensor[float]
        Cross-Entropy loss averaged by the number queries in the batch.
    """
    cross_entropy = tfl.SparseCategoricalCrossentropy(from_logits=True)
    return cross_entropy(target, output, sample_weight)


def calculate_l1_loss(target: tf.Tensor, output: tf.Tensor):
    """Calculate L1 loss for batch bounding box predictions.

    Parameters
    ----------
    target : tf.Tensor
        Ordered target bounding boxes of shape [#Objects ,4]
    output : tf.Tensor
        Ordered output bounding boxes of shape [#Objects ,4]

    Returns
    -------
    tf.Tensor[float]
        L1 loss averaged by the number of objects in the batch.
    """
    l1_loss = tf.reduce_sum(tf.math.abs(target - output), axis=1)
    return tf.reduce_mean(l1_loss)


def calculate_giou_loss(target: tf.Tensor, output: tf.Tensor):
    """Calculate generatlized IoU loss for batch bounding box predictions.

    Parameters
    ----------
    target : tf.Tensor
        Ordered target bounding boxes of shape [#Objects ,4]
    output : tf.Tensor
        Ordered output bounding boxes of shape [#Objects ,4]

    Returns
    -------
    tf.Tensor[float]
        GIoU loss averaged by the number of objects in the batch.
    """
    giou_loss = tfa.losses.giou_loss(
        y_true=box_x1y1wh_to_yxyx(target), y_pred=box_x1y1wh_to_yxyx(output)
    )
    return tf.reduce_mean(giou_loss)


def calculate_dice_loss(target: tf.Tensor, output: tf.Tensor):
    """Calculate dice loss for batch segmentation mask predictions.

    Parameters
    ----------
    target : tf.Tensor
        Ordered target masks in batch of shape [#Objects, H * W].
    output : tf.Tensor
        Ordered output segmentation masks in batch of shape [#Objects, H * W].

    Returns
    -------
    tf.Tensor[float]
        Dice loss averaged by the number of objects in the batch.
    """
    numerator = 2 * tf.reduce_sum(output * target, axis=1)
    denominator = tf.reduce_sum(target, axis=1) + tf.reduce_sum(output, axis=1)

    dice_loss = 1 - (numerator + 1) / (denominator + 1)

    return tf.reduce_mean(dice_loss)


def calculate_focal_loss(
    target: tf.Tensor, output: tf.Tensor, alpha: float = 0.25, gamma: float = 2.0
):
    """Calculate focal loss for batch segmentation mask predictions.

    Focal loss follows the proposal in https://arxiv.org/pdf/1708.02002.pdf.

    Parameters
    ----------
    target : tf.Tensor
        Ordered target masks in batch of shape [#Objects, H * W].
    output : tf.Tensor
        Ordered output segmentation masks in batch of shape [#Objects, H * W].
    alpha : float, optional
        Weighting factor to balance positive and negative examples.
    gamma : float, optional
        Exponent of modulating factor (1 - p_t) to balance easy vs. hard examples.

    Returns
    -------
    tf.Tensor[float]
        Focal loss averaged by the number of objects in the batch.
    """
    bce = K.binary_crossentropy(target, output)
    p_t = output * target + (1 - output) * (1 - target)
    focal_loss = bce * ((1 - p_t) ** gamma)

    if alpha > 0:
        alpha_t = alpha * target + (1 - alpha) * (1 - target)
        focal_loss = alpha_t * focal_loss

    return tf.reduce_mean(focal_loss)
