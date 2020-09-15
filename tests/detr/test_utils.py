import ipdb  # noqa: F401
import tensorflow as tf

from detr_models.detr.utils import box_x1y1wh_to_yxyx
from detr_models.detr.utils import extract_ordering_idx


def test_box_x1y1wh_to_yxyx():

    bboxes = [[1, 1, 5, 5], [2, 2, 4, 4]]

    expected_result = [[1, 1, 6, 6], [2, 2, 6, 6]]
    assert (box_x1y1wh_to_yxyx(bboxes).numpy() == expected_result).all()


def test_extract_ordering_idx():
    idx_slices = tf.convert_to_tensor(
        [
            [3, 4, 5, -1, -1, -1, -1, -1, -1, -1],
            [1, 9, 0, 2, -1, -1, -1, -1, -1, -1],
            [7, 8, 9, -1, -1, -1, -1, -1, -1, -1],
        ],
        dtype=tf.int64,
    )

    expected_result = [
        [0, 3],
        [0, 4],
        [0, 5],
        [1, 1],
        [1, 9],
        [1, 0],
        [1, 2],
        [2, 7],
        [2, 8],
        [2, 9],
    ]
    assert (extract_ordering_idx(idx_slices).numpy() == expected_result).all()
