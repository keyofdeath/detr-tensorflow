"""Helper functions that can be used within the RPN module
"""
import pickle

import ipdb  # noqa: F401
import numpy as np
import tensorflow as tf


def save_training_loss(rpn_loss: list, path: str):
    """Save the loss values to a .txt file

    Parameters
    ----------
    rpn_loss : list
        Training loss values
    path : str
        Path to save .txt file
    """

    with open(path, "wb") as fp:
        pickle.dump(rpn_loss, fp)


def load_training_loss(path: str):
    """Load the loss values from a .txt file

    Parameters
    ----------
    path : str
        Path to load .txt file from

    Returns
    -------
    list
        Training loss values stored in path
    """

    with open(path, "rb") as fp:
        rpn_loss = pickle.load(fp)

    return np.array(rpn_loss)


def create_bbox_mask(
    img, x_min: int, y_min: int, x_max: int, y_max: int, color=[255, 0, 255]
):
    """Create a image mask given the bounding box coordinates

    Parameters
    ----------
    img : np.array
    x_min : int
    y_min : int
    x_max : int
    y_max : int
    color : list, optional

    Returns
    -------
    np.array
        Bounding box mask
    """
    mask = np.zeros_like(img)
    mask[y_min, x_min:x_max, :] = color
    mask[y_max, x_min:x_max, :] = color
    mask[y_min:y_max, x_min, :] = color
    mask[y_min:y_max, x_max, :] = color
    return mask.astype(int)


@tf.function(input_signature=[tf.TensorSpec(shape=[None, 4], dtype=tf.float32)])
def box_cxcywh_to_xyxy(bboxes):

    num_queries = tf.shape(bboxes)[0]
    xyxy = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    for idx in range(num_queries):
        sample = bboxes[idx]
        x_c = tf.gather(sample, 0)
        y_c = tf.gather(sample, 1)
        w = tf.gather(sample, 2)
        h = tf.gather(sample, 3)
        sample = tf.stack(
            [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
        )
        xyxy = xyxy.write(idx, sample)

    return xyxy.stack()
