# IPDB can be used for debugging.
# Ignoring flake8 error code F401
import ipdb  # noqa: F401
import tensorflow as tf
import tensorflow_addons as tfa
from detr_models.detr.config import DefaultDETRConfig
from detr_models.detr.utils import box_x1y1wh_to_yxyx
from scipy.optimize import linear_sum_assignment


@tf.function
def tf_linear_sum_assignment(sample_cost_matrix, obj_indices, max_obj=tf.constant(30)):
    """Compute the bipartite linear sum assignment offered by scipy.

    Parameters
    ----------
    sample_cost_matrix : tf.Tensor
        Cost matrix of shape [#Queries, #BatchObjects].
    obj_indices : tf.Tensor
        Helper tensor used to link the object indices in the cost matrix
        to the currently considered sample.
        Padded with `-1` to transform tf.RaggedTensor to regular tf.Tensor.
    max_obj : tf.Tensor, optional
        Estimated maximum number of objects present in one sample, used for padding all
        return tensors into the same output shape.

    Returns
    -------
    tf.Tensor
        The indices of the given sample, indicating the minimal cost bipartite assignment between
        the queries and the sample object. The shape of the tensor is [2, max_obj], padded with
        `-1` where no object was present to transform tf.RaggedTensor to regular tf.Tensor.
        The first dimension corresponds to the query indices and the second dimension
        to the corresponding object indices.
    """

    sample_obj_idx = tf.gather(
        obj_indices, tf.where(tf.math.not_equal(obj_indices, -1)), axis=0
    )

    sample_cost_matrix = tf.gather(sample_cost_matrix, sample_obj_idx, axis=1)

    sample_cost_matrix = tf.reshape(
        sample_cost_matrix,
        shape=[tf.shape(sample_cost_matrix)[0], tf.shape(sample_obj_idx)[0]],
    )

    ind = tf.numpy_function(
        func=linear_sum_assignment, inp=[sample_cost_matrix], Tout=(tf.int64, tf.int64)
    )

    ind = tf.convert_to_tensor(ind, dtype=tf.int64)
    paddings = [[0, 0], [0, max_obj - tf.shape(ind)[1]]]

    return tf.pad(
        ind,
        paddings,
        "CONSTANT",
        constant_values=tf.constant(-1, dtype=tf.int64),
        name="LinearSumPadding",
    )


@tf.function
def prepare_cost_matrix(
    detr_scores,
    detr_bbox,
    batch_cls,
    batch_bbox,
    l1_cost_factor=tf.constant(5.0, dtype=tf.float32),
    iou_cost_factor=tf.constant(2.0, dtype=tf.float32),
):
    """Calculate the cost matrix of the given model outputs, required for the bipartite
    assignment between queries and objects.

    Parameters
    ----------
    detr_scores : tf.Tensor
        Batch detr score outputs of shape [Batch Size. #Queries, #Objects + 1].
    detr_bbox : tf.Tensor
        Batch detr bounding box outputs of shape [Batch Size. #Queries, 4].
    batch_cls : tf.Tensor
        Batch class targets of shape [Batch Size, #Queries, 1].
    batch_bbox : tf.Tensor
        Batch bounding box targets of shape [Batch Size, #Queries, 4].
    l1_cost_factor : tf.Tensor, optional
        Cost factor for L1-loss.
    iou_cost_factor : tf.Tensor, optional
        Cost factor for generalized IoU loss.

    Returns
    -------
    cost_matrix : tf.Tensor
        Cost matrix of the given scores in the form [Batch Size, #Queries, #BatchObjects].
        Note that the number of objects refers to the objects inside the batch. For the bipartite assignment,
        these get sliced to match only the considered sample.
    """
    config = DefaultDETRConfig()
    batch_size = tf.shape(detr_scores)[0]
    num_queries = config.num_queries
    no_object_class = config.num_classes
    num_classes = config.num_classes + 1

    # Prepare Outputs
    # [BS * #Queries , #Cls]
    # [BS * #Queries , #Coord]
    out_prob = tf.nn.softmax(
        tf.reshape(detr_scores, shape=(batch_size * num_queries, num_classes)), -1
    )
    out_bbox = tf.reshape(detr_bbox, shape=(batch_size * num_queries, 4))

    # Cls Costs
    # Objekt Klassen in Batch
    # [Class_Obj1, Class_Obj2, ...]
    obj_classes = tf.gather_nd(
        batch_cls, tf.where(tf.not_equal(batch_cls, no_object_class))
    )
    obj_classes = tf.cast(obj_classes, dtype=tf.int64)
    num_objects = tf.size(obj_classes)
    one = tf.constant(1, dtype=tf.int32)
    zero = tf.constant(0.0)
    # Compute the classification cost. Contrary to the loss, we don't use the NLL,
    # but approximate it in 1 - proba[target class].
    # The 1 is a constant that doesn't change the matching, it can be ommitted.
    cost_class = -tf.gather(out_prob, obj_classes, axis=1)
    cost_class = tf.reshape(cost_class, shape=(batch_size * num_queries, num_objects))

    # BBOX Costs
    # Objekt Bounding Boxes in Batch
    # [#Obj , #Coord]
    obj_bboxes_xywh = tf.gather_nd(
        batch_bbox, tf.where(tf.reduce_any(tf.not_equal(batch_bbox, zero), axis=-1))
    )
    obj_bboxes_xywh = tf.reshape(obj_bboxes_xywh, shape=[num_objects, 4])
    obj_bboxes_yxyx = box_x1y1wh_to_yxyx(obj_bboxes_xywh)

    # Prepare Out BBOX
    # [BS * #Queries, #Obj, #Coord]
    # 1. In Centroid Shape for L1 Loss
    out_bbox_xywh = tf.expand_dims(out_bbox, axis=1)
    out_bbox_xywh = tf.tile(out_bbox_xywh, [one, num_objects, one])
    # 2. In Positional Shape for IoU Loss
    out_bbox_yxyx = box_x1y1wh_to_yxyx(out_bbox)
    out_bbox_yxyx = tf.tile(
        tf.expand_dims(out_bbox_yxyx, axis=1), [one, num_objects, one]
    )

    # Calculate Bounding Box Matching Costs
    # L1 Loss
    cost_l1 = tf.math.reduce_sum(tf.math.abs(out_bbox_xywh - obj_bboxes_xywh), 2)

    # GIoU Loss
    giou_loss = tfa.losses.GIoULoss(reduction=tf.keras.losses.Reduction.NONE)
    cost_iou = -giou_loss(y_true=obj_bboxes_yxyx, y_pred=out_bbox_yxyx)
    cost_iou = tf.reshape(cost_iou, shape=[batch_size * num_queries, num_objects])

    cost_bbox = l1_cost_factor * cost_l1 + iou_cost_factor * cost_iou

    # Combine to Cost Matrix
    cost_matrix = cost_class + cost_bbox
    cost_matrix = tf.reshape(cost_matrix, shape=[batch_size, num_queries, num_objects])

    return cost_matrix


@tf.function
def bipartite_matching(detr_scores, detr_bbox, batch_cls, batch_bbox, obj_indices):
    """Execute the bipartite matching. Given the output scores and bounding boxes, we
    first create a cost matrix using the negative log probabilities, GIoU and L1 loss.
    Then, we assign each object the best fitting (minimal cost) query.

    Parameters
    ----------
    detr_scores : tf.Tensor
        Batch detr score outputs of shape [Batch Size. #Queries, #Objects + 1].
    detr_bbox : tf.Tensor
        Batch detr bounding box outputs of shape [Batch Size. #Queries, 4].
    batch_cls : tf.Tensor
        Batch class targets of shape [Batch Size, #Queries, 1].
    batch_bbox : tf.Tensor
        Batch bounding box targets of shape [Batch Size, #Queries, 4].

    Returns
    -------
    tf.Tensor
        Batch indices indicating the assignement between queries and objects. The shape of
        the tensor is [Batch Size, 2, max_obj].
        The second dimension corresponds to Query_IDX, Object_IDX of the corresponding batch. For each sample,
        the indices got padded with `-1` to match `max_obj` in order to constitute a regular tensor.
    """

    cost_matrix = prepare_cost_matrix(detr_scores, detr_bbox, batch_cls, batch_bbox)

    batch_idx = tf.map_fn(
        lambda elems: tf_linear_sum_assignment(*elems),
        elems=[cost_matrix, obj_indices.to_tensor(-1)],
        dtype=tf.int64,
    )

    return batch_idx
