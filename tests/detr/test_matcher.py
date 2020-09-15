import ipdb  # noqa: F401
import numpy as np

from constants import (
    DETR_SCORES_1,
    DETR_BBOX_1,
    BATCH_CLS_1,
    BATCH_BBOX_1,
    OBJ_INDICES_1,
)
from constants import (
    DETR_SCORES_2,
    DETR_BBOX_2,
    BATCH_CLS_2,
    BATCH_BBOX_2,
    OBJ_INDICES_2,
)

from detr_models.detr.matcher import bipartite_matching


def test_bipartite_matching():

    expected_shape = (1, 2, 30)
    expected_result = np.full((2, 30), -1)
    expected_result[:, 0] = [0, 0]

    result = bipartite_matching(
        DETR_SCORES_1, DETR_BBOX_1, BATCH_CLS_1, BATCH_BBOX_1, OBJ_INDICES_1
    )

    assert expected_shape == result.shape
    assert (expected_result == result.numpy()[0]).all()

    expected_result = np.full((2, 30), -1)
    expected_result[:, 0] = [0, 1]
    expected_result[:, 1] = [1, 0]

    result = bipartite_matching(
        DETR_SCORES_2, DETR_BBOX_2, BATCH_CLS_2, BATCH_BBOX_2, OBJ_INDICES_2
    )
    assert expected_shape == result.shape
    assert (expected_result == result.numpy()[0]).all()
