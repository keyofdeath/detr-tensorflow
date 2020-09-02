# IPDB can be used for debugging.
# Ignoring flake8 error code F401
import ipdb  # noqa: F401
import numpy as np
import tensorflow as tf
from detr_models.data_feeder.uuid_iterator import UUIDIterator
from detr_models.detr.utils import progress_bar
from numba import float32, int32, jit, njit, types
from numba.typed import List
from numpy import loadtxt
from tensorflow.keras.preprocessing.image import img_to_array, load_img


class PVOCFeeder:
    """Helper Class to provide the DETR model with the training data in the required shapes."""

    def __init__(
        self, storage_path: str, num_queries: int, num_classes: int, batch_size: int
    ):
        """Initialize Data Feeder

        Parameters
        ----------
        storage_path : str
            Path to data storage.
        num_queries : int
            Number of queries used in transformer network.
        num_classes : int
            Number of target classes.
        """
        self.name = "PVOCFeeder"
        self.storage_path = storage_path
        self.num_queries = np.int32(num_queries)
        self.num_classes = np.int32(num_classes)
        self.uuid_iterator = UUIDIterator(
            storage_path=storage_path, batch_size=batch_size
        )

    def __call__(self, verbose: bool):
        """Yield batch input for DETR model.

        Parameters
        ----------
        verbose : bool
            Indicating extended logging with progress_bar.

        Yields
        ------
        input_data : list
            Input data for DETR model as specified in `prepare_inputs()`.
        """

        batches = self.uuid_iterator()
        number_batches = len(batches)
        for batch_iteration, batch_uuids in enumerate(batches):
            input_data = self.prepare_inputs(batch_uuids)
            yield input_data

            if verbose:
                progress_bar(total=number_batches, current=batch_iteration + 1)

    def prepare_inputs(self, batch_uuids: list):
        """Prepare input data for DETR.

        Parameters
        ----------
        batch_uuids : list
            List of uuids of images in `storage_path`.

        Returns
        -------
        batch_inputs : tf.Tensor
            Batch input images of shape [Batch Size, H, W, C].
        batch_cls : tf.Tensor
            Batch class targets of shape [Batch Size, #Queries, 1].
        batch_bbox : tf.Tensor
            Batch bounding box targets of shape [Batch Size, #Queries, 4].
        obj_indices : tf.RaggedTensor
            Helper tensor of shape [Batch Size, None].
            Used to link objects in the cost matrix to the target tensors.
        """
        batch_inputs, batch_labels = self.load_data(batch_uuids)

        batch_cls = []
        batch_bbox = []
        batch_size = len(batch_uuids)

        for idx in np.arange(0, batch_size, dtype=np.int32):
            sample_labels = np.array(batch_labels[idx], dtype=np.float32)

            sample_cls, sample_bbox = labels_to_targets(
                sample_labels, self.num_queries, self.num_classes
            )

            batch_cls.append(sample_cls)
            batch_bbox.append(sample_bbox)

        obj_indices = retrieve_obj_indices(np.array(batch_cls), self.num_classes)
        obj_indices = tf.ragged.constant(obj_indices, dtype=tf.int64)

        batch_inputs = tf.convert_to_tensor(batch_inputs, dtype=tf.float32)
        batch_cls = tf.convert_to_tensor(batch_cls, dtype=tf.float32)
        batch_bbox = tf.convert_to_tensor(batch_bbox, dtype=tf.float32)

        return (batch_inputs, batch_cls, batch_bbox, obj_indices)

    def load_data(self, batch_uuids: list):
        """Load the images and labels of the corresponding uuids.

        Parameters
        ----------
        batch_uuids : list
            Description

        Returns
        -------
        batch_inputs : np.array
            Batch images of shape [Batch Size, H, W, C]
        batch_labels : np.array
            Batch labels of shape [Batch Size, #Objects, 5].
            The array is nested, such that each sample can have varying number of objects.
            Further, the last dimension speciefies the cls (1) and the coordinates (4).
        """

        batch_inputs = []
        batch_labels = []

        for sample_uuid in batch_uuids:
            img = self.loadimage(sample_uuid)
            label = self.loadlabel(sample_uuid)

            batch_inputs.append(img)
            batch_labels.append(label)

        return np.array(batch_inputs), np.array(batch_labels)

    def loadimage(self, uuid: str):
        """Load and return the image given the specified uuid.

        Parameters
        ----------
        uuid : str
            UUID of a given sample

        Returns
        -------
        np.array
            Image of shape [H, W]

        """
        image_path = "{}/images/{}.jpg".format(self.storage_path, uuid)
        return img_to_array(load_img(image_path), dtype=np.float32)

    def loadlabel(self, uuid: str):
        """Load and return the labels given the specified uuid. The label contains the
        classes and the coordinates of the bounding boxes and the length of the labels corresponds
        to the number of objects in the image.

        Parameters
        ----------
        uuid : str
            UUID of a given sample

        Returns
        -------
        np.array
            Labels of shape [#Objects, 5]
        """

        label_path = "{}/labels/{}.txt".format(self.storage_path, uuid)
        labels = loadtxt(label_path, comments="#", delimiter=" ", unpack=False)
        if labels.ndim == 1:
            return labels.reshape((1, 5))
        return np.array(labels, dtype=np.float32)


@jit(types.UniTuple(float32[:, :], 2)(float32[:, :], int32, int32))
def labels_to_targets(sample_labels, num_queries, num_classes):
    """Prepare the true target class and bounding boxes to be aligned with detr output.

        Important information regarding Input
        -------------------------------------
            * Each row in the labels consists of [class, x_center, y_center, width, height]
            * Coordinates are normalized by image width (x,w) and image height (y,h)
            * Class numbers start by 0


        Parameters
        ----------
        sample_labels : list
            Ground truth/Labeled objects corresponding to the given image
        num_queries : int
            num_queriesumber of detections per image

        Returns
        -------
        sample_cls : np.array
            Ground truth class labels of the image in shape [num_queries], padded with `num_classes`.
        sample_bbox : np.array
            Ground truth bounding boxes of the image in shape [num_queries, 4] in centroid format
            [class, x_center, y_center, width, height]

        """
    sample_cls = np.full(
        shape=(num_queries, 1), fill_value=num_classes, dtype=np.float32
    )
    sample_bbox = np.full(
        shape=(num_queries, 4), fill_value=num_classes, dtype=np.float32
    )

    for idx, labeled_obj in enumerate(sample_labels):
        cls_label, x, y, w, h = labeled_obj
        sample_cls[idx] = cls_label
        sample_bbox[idx, :] = x, y, w, h

    return sample_cls, sample_bbox


@njit
def retrieve_obj_indices(batch_cls, num_classes):
    """Helper function to save the object indices for later.
    E.g. a batch of 3 samples with varying number of objects (1, 3, 1) will
    produce a mapping [[0], [1,2,3], [4]]. This will be needed later on in the
    bipartite matching.

    Parameters
    ----------
    batch_cls : np.array
        Batch class targets of shape [Batch Size, #Queries, 1].
    num_classes : int
        Number of target classes.

    Returns
    -------
    np.array
        Object indices indicating for each sample at which position the
        associated objects are.
    """
    obj_indices = List()
    batch_size = batch_cls.shape[0]

    for idx in np.arange(0, batch_size, dtype=np.int32):
        sample = batch_cls[idx]
        object_indices = np.where(sample != num_classes)[0]
        num_objects_in_sample = len(object_indices)

        if idx == 0:
            sample_obj_indices = np.arange(0, num_objects_in_sample, dtype=np.int32)
            obj_indices.append(sample_obj_indices)
            last_num_objects = num_objects_in_sample
        else:
            start, upto = last_num_objects, last_num_objects + num_objects_in_sample
            sample_obj_indices = np.arange(start, upto, dtype=np.int32)
            obj_indices.append(sample_obj_indices)
            last_num_objects = upto

    return obj_indices
