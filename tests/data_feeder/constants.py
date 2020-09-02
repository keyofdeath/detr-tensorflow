import os
from PIL import Image

COCO_PATH = "tests/data/coco.json"
IMAGE_PATH = "tests/data/images/"

TEST_IMAGE_PATH = os.path.join(IMAGE_PATH, "dog.jpg")
TEST_IMAGE = Image.open(TEST_IMAGE_PATH)
