import os


class DETRTrainingConfig:
    def __init__(self):
        # Data Storage
        self.storage_path = (
            "/Users/Schlendrikovic/Documents/Arbeit/Auvisus/Data/Detector/test"
        )
        self.count_images = len(os.listdir(self.storage_path + "/images"))

        # Input Size (H,W,C)
        self.input_shape = (270, 480, 3)

        # Number of Predictions per Image
        self.num_queries = 100

        # Number of Classes
        # Currently: Gka
        self.num_classes = 4

        # Number of heads in each attention layer
        self.num_heads = 8

        # Transformer
        self.dim_transformer = 256
        self.dim_feedforward = 2048
        self.num_transformer_layer = 6

        # Training
        self.epochs = 150
        self.batch_size = 20
        self.learning_rate = 1e-4

        # Cost Factors
        self.bbox_cost_factor = 5
        self.iou_cost_factor = 2

        # Backbone
        self.backbone_name = "ResNet50"
        self.backbone_config = {
            "input_shape": self.input_shape,
            "include_top": False,
            "weights": "imagenet",
        }
        self.train_backbone = False

        # GPU or CPU
        self.run_on_gpu = True
