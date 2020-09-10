# IPDB can be used for debugging.
# Ignoring flake8 error code F401
import os

import ipdb  # noqa: F401import os
import numpy as np
import PIL
import pytest
from constants import COCO_PATH, IMAGE_PATH, TEST_IMAGE
from detr_models.data_feeder.coco_feeder import (
    load_coco,
    load_image,
    load_target_cls_and_masks,
    padd_target_for_queries,
    resize_and_pad_image,
    resize_and_pad_masks,
    retrieve_normalized_bbox,
    retrieve_obj_indices,
    slize_batches,
)
from pycocotools.coco import COCO


def test_load_coco():
    coco = load_coco(COCO_PATH)
    assert isinstance(coco, COCO)


def test_slize_batches():
    image_ids = [0, 1, 2, 3]

    batch_size = 1
    expected_result = [[0], [1], [2], [3]]
    assert (expected_result == slize_batches(image_ids, batch_size)).all()

    batch_size = 2
    expected_result = [[0, 1], [2, 3]]
    assert (expected_result == slize_batches(image_ids, batch_size)).all()

    batch_size = 3
    expected_result = [[0, 1, 2]]
    assert (expected_result == slize_batches(image_ids, batch_size)).all()


def test_slice_batches_invalid_large_batch_size():
    image_ids = [0, 1, 2, 3]

    with pytest.raises(ValueError):
        batch_size = 5
        slize_batches(image_ids, batch_size)


def test_slice_batches_invalid_argument_image_ids():
    with pytest.raises(TypeError):
        image_ids = "not a list"
        slize_batches(image_ids)


def test_slice_batches_invalid_argument_batch_size():
    image_ids = [0, 1, 2, 3]

    with pytest.raises(TypeError):
        batch_size = "not an integer"
        slize_batches(image_ids, batch_size)


def test_load_image():
    image_path = os.path.join(IMAGE_PATH, "cat.jpg")
    image = load_image(image_path)
    assert isinstance(image, PIL.JpegImagePlugin.JpegImageFile)
    assert image.size == (1024, 1365)


def test_load_image_file_not_found():
    with pytest.raises(FileNotFoundError):
        image_path = "no-image-here.jpg"
        load_image(image_path)


def test_load_image_invalid_argument():
    with pytest.raises(AttributeError):
        image_path = 1
        load_image(image_path)


def test_load_target_cls_and_masks():
    coco = load_coco(COCO_PATH)
    result = load_target_cls_and_masks(coco, img_id=4)
    expected_label = [2, 2]

    assert len(result) == 2
    assert result[0].shape == (2,)
    assert (expected_label == result[0]).all()

    expected_mask_shapes = [(768, 1024), (768, 1024)]
    assert [mask.shape for mask in result[1]] == expected_mask_shapes


def test_load_target_cls_and_masks_invalid_argument():
    with pytest.raises(TypeError):
        coco = load_coco(COCO_PATH)
        load_target_cls_and_masks(coco, img_id="not an integer")


def test_padd_target_for_queries():
    target = np.array([1, 2, 3, 1])
    num_columns = 1
    num_queries = 5
    num_classes = 4

    result = padd_target_for_queries(target, num_columns, num_queries, num_classes)
    expected_result = [[1], [2], [3], [1], [4]]
    assert result.shape == (num_queries, num_columns)
    assert (result == expected_result).all()

    target = np.array([[1, 1, 1, 1], [1, 2, 3, 1], [2, 2, 2, 2], [1, 2, 3, 1]])
    num_columns = 4

    result = padd_target_for_queries(target, num_columns, num_queries, num_classes)
    expected_result = [
        [1, 1, 1, 1],
        [1, 2, 3, 1],
        [2, 2, 2, 2],
        [1, 2, 3, 1],
        [4, 4, 4, 4],
    ]
    assert result.shape == (num_queries, num_columns)
    assert (result == expected_result).all()


def test_padd_target_for_queries_too_many_objects():
    target = np.array([1, 2, 3, 1])
    num_columns = 1
    num_queries = 3
    num_classes = 4

    with pytest.raises(ValueError):
        padd_target_for_queries(target, num_columns, num_queries, num_classes)


def test_retrieve_obj_indices():
    batch_cls = np.array([[1, 4, 4, 4], [1, 2, 4, 4]])
    num_classes = 4

    result = retrieve_obj_indices(batch_cls, num_classes)
    expected_result = [[0], [1, 2]]
    assert result == expected_result

    batch_cls = np.array([[1, 1, 1, 4], [1, 2, 4, 4]])
    num_classes = 4

    result = retrieve_obj_indices(batch_cls, num_classes)
    expected_result = [[0, 1, 2], [3, 4]]
    assert result == expected_result


def test_resize_and_pad_image():
    image_width = int(TEST_IMAGE.width * 0.5)
    image_height = int(TEST_IMAGE.height * 0.5)

    result = resize_and_pad_image(TEST_IMAGE, image_width, image_height)
    expected_shape = (image_height, image_width, 3)
    assert isinstance(result, np.ndarray)
    assert result.shape == expected_shape


def test_resize_and_pad_masks():
    image_width = int(TEST_IMAGE.width * 0.5)
    image_height = int(TEST_IMAGE.height * 0.5)

    coco = load_coco(COCO_PATH)
    _, sample_masks = load_target_cls_and_masks(coco, img_id=2)

    result = resize_and_pad_masks(sample_masks, image_width, image_height)
    expected_shape = [(image_height, image_width)]
    assert [mask.shape for mask in result] == expected_shape


def test_retrieve_normalized_bbox():
    mask = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )
    height, width = 5, 6
    result = retrieve_normalized_bbox([mask], width, height)
    expected_result = (1 / width, 1 / height, 3 / width, 2 / height)
    assert len(result[0]) == 4
    assert (result[0] == expected_result).all()

    mask = np.array(
        [
            [0, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 0],
        ]
    )

    result = retrieve_normalized_bbox([mask], width, height)
    expected_result = (1 / width, 0 / height, 3 / width, 4 / height)
    assert len(result[0]) == 4
    assert (result[0] == expected_result).all()
