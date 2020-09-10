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
def box_x1y1wh_to_yxyx(bboxes):

    num_queries = tf.shape(bboxes)[0]
    yxyx = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    for idx in range(num_queries):
        sample = bboxes[idx]
        x1 = tf.gather(sample, 0)
        y1 = tf.gather(sample, 1)
        w = tf.gather(sample, 2)
        h = tf.gather(sample, 3)
        sample = tf.stack([y1, (x1), (y1 + h), (x1 + w)])
        yxyx = yxyx.write(idx, sample)

    return yxyx.stack()


def create_positional_encodings(fm_shape, num_pos_feats, batch_size):
    """Helper function to create the positional encodings used in the
    transformer network of sinus type.

    Parameters
    ----------
    fm_shape : tuple
        Shape of feature map used to create positional encodings [H,W].
    num_pos_feats : int
        Number of dimensions to express each position in. As both the x and y
        coordinate is expressed in `num_pos_feats` dimensions and then added,
        this number should be 0.5 * dim_transformer.
    batch_size : int

    Returns
    -------
    tf.Tensor
            Positional encodings of shape [Batch Size, H*W, dim_transformer].
            Used in transformer network to enrich input information.
    """
    height, width, c = fm_shape

    y_embed = np.repeat(np.arange(1, height + 1), width).reshape(height, width)
    x_embed = np.full(shape=(height, width), fill_value=np.arange(1, width + 1))

    # d/2 entries for each dimension x and y
    div_term = np.arange(num_pos_feats)
    div_term = 10000 ** (2 * (div_term // 2) / num_pos_feats)
    pos_x = x_embed[:, :, None] / div_term
    pos_y = y_embed[:, :, None] / div_term

    pos_x_even = np.sin(pos_x[:, :, 0::2])
    pos_x_uneven = np.sin(pos_x[:, :, 1::2])

    pos_y_even = np.sin(pos_y[:, :, 0::2])
    pos_y_uneven = np.sin(pos_y[:, :, 1::2])

    pos_x = np.concatenate([pos_x_even, pos_x_uneven], axis=2)
    pos_y = np.concatenate([pos_y_even, pos_y_uneven], axis=2)

    positional_encodings = np.concatenate([pos_y, pos_x], axis=2)
    positional_encodings = np.expand_dims(positional_encodings, 0)
    positional_encodings = np.repeat(positional_encodings, batch_size, axis=0)

    positional_encodings = tf.convert_to_tensor(positional_encodings, dtype=tf.float32)
    positional_encodings = tf.reshape(
        positional_encodings,
        shape=(batch_size, height * width, positional_encodings.shape[3]),
    )

    return positional_encodings


def progress_bar(total: int, current: int):
    """Helper function to plot progress bar."""
    percentage = int((current / total) * 100)

    if percentage != 0:
        pstr = "Progress: "

        for _ in range(0, percentage // 10):
            pstr = "{}{}".format(pstr, "-")

        pstr = "{} {}%".format(pstr, percentage)
        print(pstr, end="\r", flush=True)
