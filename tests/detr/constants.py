import numpy as np
import tensorflow as tf

DETR_SCORES_1 = [[0.25, 0.75, 0.25], [0.25, 0.5, 0.75], [0.25, 0.5, 0.7]]
DETR_SCORES_1 = np.expand_dims(DETR_SCORES_1, 0)
DETR_SCORES_1 = tf.convert_to_tensor(DETR_SCORES_1, dtype=tf.float32)

DETR_BBOX_1 = [[0.1, 0.1, 0.5, 0.5], [0.5, 0.5, 0.1, 0.1], [0.7, 0.7, 0.1, 0.1]]
DETR_BBOX_1 = np.expand_dims(DETR_BBOX_1, 0)
DETR_BBOX_1 = tf.convert_to_tensor(DETR_BBOX_1, dtype=tf.float32)

BATCH_CLS_1 = [[1], [2], [2]]
BATCH_CLS_1 = np.expand_dims(BATCH_CLS_1, 0)
BATCH_CLS_1 = tf.convert_to_tensor(BATCH_CLS_1, dtype=tf.float32)


BATCH_BBOX_1 = [[0.1, 0.1, 0.5, 0.5], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
BATCH_BBOX_1 = np.expand_dims(BATCH_BBOX_1, 0)
BATCH_BBOX_1 = tf.convert_to_tensor(BATCH_BBOX_1, dtype=tf.float32)


OBJ_INDICES_1 = [0]
OBJ_INDICES_1 = np.expand_dims(OBJ_INDICES_1, 0)
OBJ_INDICES_1 = tf.ragged.constant(OBJ_INDICES_1, dtype=tf.int64)


# More difficult
DETR_SCORES_2 = [[2, 0.1, 0.25], [0.25, 0.9, 0.25], [0.1, 0.1, 2]]
DETR_SCORES_2 = np.expand_dims(DETR_SCORES_2, 0)
DETR_SCORES_2 = tf.convert_to_tensor(DETR_SCORES_2, dtype=tf.float32)

DETR_BBOX_2 = [[0.5, 0.6, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1], [0.6, 0.6, 0.1, 0.1]]
DETR_BBOX_2 = np.expand_dims(DETR_BBOX_2, 0)
DETR_BBOX_2 = tf.convert_to_tensor(DETR_BBOX_2, dtype=tf.float32)

BATCH_CLS_2 = [[1], [0], [2]]
BATCH_CLS_2 = np.expand_dims(BATCH_CLS_2, 0)
BATCH_CLS_2 = tf.convert_to_tensor(BATCH_CLS_2, dtype=tf.float32)


BATCH_BBOX_2 = [[0.1, 0.1, 0.2, 0.3], [0.6, 0.6, 0.2, 0.3], [0.0, 0.0, 0.0, 0.0]]
BATCH_BBOX_2 = np.expand_dims(BATCH_BBOX_2, 0)
BATCH_BBOX_2 = tf.convert_to_tensor(BATCH_BBOX_2, dtype=tf.float32)


OBJ_INDICES_2 = [0, 1]
OBJ_INDICES_2 = np.expand_dims(OBJ_INDICES_2, 0)
OBJ_INDICES_2 = tf.ragged.constant(OBJ_INDICES_2, dtype=tf.int64)
