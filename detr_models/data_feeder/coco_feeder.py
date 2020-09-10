# IPDB can be used for debugging.
# Ignoring flake8 error code F401
import io
import os
import random
from contextlib import redirect_stdout

import ipdb  # noqa: F401
import numpy as np
import tensorflow as tf
from detr_models.detr.utils import progress_bar
from PIL import Image
from pycocotools.coco import COCO


class COCOFeeder:
    """Helper Class to provide the DETR model with the training data in the required shapes."""

    def __init__(
        self,
        storage_path: str,
        num_queries: int,
        num_classes: int,
        batch_size: int,
        image_width: int,
        image_height: int,
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
        batch_size : int
            Number of samples in batch.
        width : int
            Image width used for scaling of input images and masks.
        height : int
            Image height used for scaling of input images and masks.
        """
        self.name = "COCOFeeder"

        self.storage_path = storage_path
        print("\nFeed DETR model with data from: {}\n".format(storage_path), flush=True)

        self.image_path = os.path.join(storage_path, "images")

        self.num_queries = np.int32(num_queries)
        self.num_classes = np.int32(num_classes)
        self.batch_size = batch_size

        self.image_height = image_height
        self.image_width = image_width

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
        path_to_coco = [
            os.path.join(self.storage_path, file)
            for file in os.listdir(self.storage_path)
            if ".json" in file
        ][0]
        coco = load_coco(path_to_coco)

        image_ids = coco.getImgIds()

        image_ids = slize_batches(image_ids, self.batch_size)
        # Each  Iteration consists of k=1000 steps
        image_ids = random.choices(image_ids, k=1000)

        for batch_iteration, batch_image_ids in enumerate(image_ids):
            try:
                input_data = self.prepare_inputs(coco, batch_image_ids)
            except Exception as ex:
                print("Could not Read Image IDs: `{}` - {}".format(batch_image_ids, ex))
                continue

            if verbose:
                print("Read Batch Imgs: {}".format(batch_image_ids))
                print("Batch Images Input Shape:{}".format(input_data[0].shape))
                print("Batch Class Labels Input Shape:{}".format(input_data[1].shape))
                print(
                    "Batch Class Bounding Boxes Input Shape:{}".format(
                        input_data[2].shape
                    )
                )
                print("Object IDs matching to Image ID: {}".format(input_data[3]))

            yield input_data

            progress_bar(total=len(image_ids), current=batch_iteration + 1)

    def prepare_inputs(self, coco: object, batch_image_ids: list):
        """Prepare input data for DETR.

        Parameters
        ----------
        batch_uuids : list
            List of uuids of images in `storage_path`.

        Returns
        -------
        batch_images : tf.Tensor
            Batch input images of shape [Batch Size, Height, Width, 3].
        batch_cls : tf.Tensor
            Batch class targets of shape [Batch Size, #Queries, 1].
        batch_bboxs : tf.Tensor
            Batch bounding box targets of shape [Batch Size, #Queries, 4].
        batch_masks : tf.Tensor
            Batch segmentation masks of shape [Batch size, #Objects, Height, Width]
        obj_indices : tf.RaggedTensor
            Helper tensor of shape [Batch Size, None].
            Used to link objects in the cost matrix to the target tensors.
        """

        batch_images = []
        batch_cls = []
        batch_bboxs = []
        batch_masks = []
        for img_id in batch_image_ids:
            # Load image
            img_dict = coco.loadImgs([img_id])[0]
            image_path = os.path.join(
                self.storage_path, "images", img_dict.get("file_name")
            )

            sample_image = load_image(image_path)
            sample_image = resize_and_pad_image(
                sample_image, width=self.image_width, height=self.image_height
            )

            # Load target class and masks
            sample_cls, sample_masks = load_target_cls_and_masks(coco, img_id)
            sample_masks = resize_and_pad_masks(
                sample_masks, width=self.image_width, height=self.image_height
            )

            # Retrieve Bounding Boxes
            sample_bboxs = retrieve_normalized_bbox(
                sample_masks=sample_masks,
                width=self.image_width,
                height=self.image_height,
            )

            # Padd target classes and bounding boxes
            sample_cls = padd_target_for_queries(
                target=sample_cls,
                num_columns=1,
                num_queries=self.num_queries,
                fill_value=self.num_classes,
            )
            sample_bboxs = padd_target_for_queries(
                target=sample_bboxs,
                num_columns=4,
                num_queries=self.num_queries,
                fill_value=0,
            )

            # Append to placeholders
            batch_images.append(sample_image)
            batch_cls.append(sample_cls)
            batch_bboxs.append(sample_bboxs)
            batch_masks.append(sample_masks)

        batch_images = np.array(batch_images)
        batch_cls = np.array(batch_cls)
        batch_bboxs = np.array(batch_bboxs)
        batch_masks = np.array(batch_masks)

        batch_images = tf.convert_to_tensor(batch_images, dtype=tf.float32)
        batch_cls = tf.convert_to_tensor(batch_cls, dtype=tf.float32)
        batch_bboxs = tf.convert_to_tensor(batch_bboxs, dtype=tf.float32)

        # Batch masks to tensor
        # Each sampel can have multiple different
        batch_masks = tf.ragged.constant(batch_masks, dtype=tf.float32)

        # Get Object Indices for each sample in batch
        obj_indices = retrieve_obj_indices(batch_cls, self.num_classes)
        obj_indices = tf.ragged.constant(obj_indices, dtype=tf.int64)

        return (batch_images, batch_cls, batch_bboxs, obj_indices, batch_masks)


def load_coco(path_to_coco: str):
    """Load coco object given the specified path to the COCO annotation file.

    Parameters
    ----------
    path_to_coco : str
        Path to COCO annotation file.

    Returns
    -------
    coco : object
        COCO object as specified by `https://github.com/cocodataset/cocoapi/tree/master/PythonAPI`
    """
    with redirect_stdout(io.StringIO()):
        return COCO(path_to_coco)


def slize_batches(image_ids: list, batch_size: int = 1):
    """Slize a given list of image ids into batches of equal size.

    Parameters
    ----------
    image_ids : list
    batch_size : int, optional

    Returns
    -------
    image_ids : np.ndarray
        Sliced list of image_ids with each entry of size `batch_size`.
    """
    if batch_size > len(image_ids):
        raise ValueError(
            "Batch Size to large for image ids of size `{}`".format(len(image_ids))
        )

    if not isinstance(image_ids, list):
        raise TypeError("Image IDs not in correct format `{}`".format(type(image_ids)))

    if batch_size > 1:
        num_batches = len(image_ids) // batch_size
        image_ids = [
            image_ids[(_ * batch_size) : (_ * batch_size + batch_size)]
            for _ in range(num_batches)
        ]
    else:
        image_ids = [[image_id] for image_id in image_ids]
    return np.array(image_ids)


def load_image(image_path: str):
    """Load a specified image.

    Parameters
    ----------
    image_path : str

    Returns
    -------
    image : PIL.JpegImagePlugin.JpegImageFile
    """
    return Image.open(image_path)


def load_target_cls_and_masks(coco: object, img_id: int):
    """Load the target labels from COCO for a given image specified by `img_id`.

    Note
    ----
    In cae the annotation file does not start with category_id 0, we subtract the `min_category`
    from the cls_labels in order to align with standard python indexing starting at `0`.

    Parameters
    ----------
    coco : object
        Imported COCO object.
    img_id : int
        Image id used to load targets from.

    Returns
    -------
    sample_cls : np.ndarray
        True class labels for the given image of shape (#Objects, ).
    sample_masks : np.ndarray
        True masks for the given image of shape (#Objects, Height, Width).
    """

    if not isinstance(img_id, (int, np.integer)):
        raise TypeError("Image ID not in correct format `{}`".format(type(img_id)))

    sample_cls = []
    sample_masks = []

    min_category = np.min(coco.getCatIds())

    anns = coco.loadAnns(ids=coco.getAnnIds(imgIds=img_id))

    for ann in anns:
        label = ann.get("category_id") - min_category
        mask = coco.annToMask(ann)

        sample_cls.append(label)
        sample_masks.append(mask)

    return np.array(sample_cls), np.array(sample_masks)


def _downscale_image(image: Image, width: int, height: int):
    """Downscale image to match smallest size.

    The scaling factor can be derived by taking the minimum of both possible
    factors:

        factor = min{ (width/original_width), (height/original_height) }

    Parameters
    ----------
    image : PIL.Image
    width : int
        Resized image width.
    height : int
        Resized image height.

    Returns
    -------
    image : PIL.Image
        Image resized to shape (original_height*factor, original_width*factor).
    """
    factor = np.min([width / image.width, height / image.height])
    size = (int(image.width * factor), int(image.height * factor))

    return image.resize(size=size)


def _pad_image(image: Image, width: int, height: int):
    """Pad a given image to match specified widht and height.

    Parameters
    ----------
    image : PIL.Image
    width : int
        Resized image width.
    height : int
        Resized image height.

    Returns
    -------
    padded_image : PIL.Image
        Image padded to shape (height, width).
    """
    padded_image = Image.new(image.mode, (width, height))
    upper_left = ((width - image.width) // 2, (height - image.height) // 2)
    padded_image.paste(image, box=upper_left)

    return padded_image


def resize_and_pad_image(image: Image, width: int, height: int):
    """Resize a PIL Image to match specified shape.

    Parameters
    ----------
    image : PIL.Image
    width : int
        Resized image width.
    height : int
        Resized image height.

    Returns
    -------
    image : np.ndarray
        Resized image of shape [Height, Width, 3]
    """
    if image.size != (width, height):
        image = _downscale_image(image, width, height)

        if (image.width, image.height) != (width, height):
            image = _pad_image(image, width, height)

    return np.asarray(image)


def resize_and_pad_masks(sample_masks: np.ndarray, width: int, height: int):
    """Resize masks to match specified shape.

    Parameters
    ----------
    sample_masks : np.ndarray
    width : int
        Resized image width.
    height : int
        Resized image height.

    Returns
    -------
    masks : np.ndarray
        Resized masks of shape [#Objects, Height, Width]
    """
    sample_masks = [Image.fromarray(mask) for mask in sample_masks]
    sample_masks = [resize_and_pad_image(mask, width, height) for mask in sample_masks]
    return np.array(sample_masks)


def retrieve_normalized_bbox(sample_masks: np.ndarray, width: int, height: int):
    """Retrieve the bounding box from the resized segmentation mask and normalize the
    coordinates using the image shapes.

    Parameters
    ----------
    sample_masks : np.ndarray
        Resized masks of shape [#Objects, Height, Width]
    width : int
        Resized image width.
    height : int
        Resized image height.

    Returns
    -------
    sample_bboxs : np.ndarray
        Normalized bounding boxes of shape [#Objects, 4].
        The coordinates are specified as [xmin, ymin, width, height].
    """
    sample_bboxs = []

    for mask in sample_masks:
        ycoord, xcoord = np.where(mask != 0)
        ymin, ymax = np.min(ycoord), np.max(ycoord)
        xmin, xmax = np.min(xcoord), np.max(xcoord)
        w, h = (xmax - xmin), (ymax - ymin)
        bbox = xmin / width, ymin / height, w / width, h / height

        sample_bboxs.append(bbox)

    return np.array(sample_bboxs)


def padd_target_for_queries(
    target: np.ndarray, num_columns: int, num_queries: int, fill_value: int
):
    """Padd a given target to match number of queries.

    Usually, #Objects < #Queries and we therefore need to padd the targets to match the
    predictions in the model. We use `fill_value` to fill the arrays.

    Parameters
    ----------
    target : np.ndarray
        Target array of shape (#Objects, num_columns)
    num_columns : int
        Number of columns for padding.
    num_queries : int
        Number of queries used in transformer network.
    fill_value : int
        Padding fill value.

    Returns
    -------
    padded : np.ndarray
        Padded target array of shape (#Queries, num_columns), padded with `fill_value`.
    """

    if target.shape[0] > num_queries:
        raise ValueError(
            "More objects in image than queries specified. Increase the number of queries!"
        )

    padded = np.full(
        shape=(num_queries, num_columns), fill_value=fill_value, dtype=np.float32
    )
    for idx, target_object in enumerate(target):
        padded[idx, :] = target_object

    return padded


def retrieve_obj_indices(batch_cls: np.ndarray, num_classes: int):
    """Helper function to save the object indices for later.
    E.g. a batch of 3 samples with varying number of objects (1, 3, 1) will
    produce a mapping [[0], [1,2,3], [4]]. This will be needed later on in the
    bipartite matching.

    Parameters
    ----------
    batch_cls : np.ndarray
        Batch class targets of shape [Batch Size, #Queries, 1].
    num_classes : int
        Number of target classes.

    Returns
    -------
    obj_indices : list
        Object indices indicating for each sample at which position the
        associated objects are.
    """
    obj_indices = []
    batch_size = batch_cls.shape[0]

    for idx in np.arange(0, batch_size, dtype=np.int32):
        sample = batch_cls[idx]
        object_indices = np.where(sample != num_classes)[0]
        num_objects_in_sample = len(object_indices)

        if idx == 0:
            sample_obj_indices = np.arange(0, num_objects_in_sample, dtype=np.int32)
            obj_indices.append(sample_obj_indices.tolist())
            last_num_objects = num_objects_in_sample
        else:
            start, upto = last_num_objects, last_num_objects + num_objects_in_sample
            sample_obj_indices = np.arange(start, upto, dtype=np.int32)
            obj_indices.append(sample_obj_indices.tolist())
            last_num_objects = upto

    return obj_indices
