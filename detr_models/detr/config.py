class DefaultDETRConfig:
    def __init__(self):
        # Specify Data Type
        # Currently only COCO or PVOC
        self.data_type = "COCO"

        # Number of Predictions per Image
        self.num_queries = 60

        # Number of Classes in the dataset
        # Please note that the actual numbering of the classes
        # should start at zero e.g. 0, 1, 2, 3 in the default case.
        self.num_classes = 4

        # Transformer Config
        self.dim_transformer = 256
        self.dim_feedforward = 512
        self.num_transformer_layer = 2
        self.num_heads = 8

        # Training Config
        self.epochs = 300
        self.batch_size = 1
        self.learning_rate = 1e-4
        self.drops = [100, 200]
        self.weight_decay = 1e-4

        # Backbone Config
        # MobileNetV2, ResNet50
        self.backbone_name = "MobileNetV2"
        self.train_backbone = True

        self.train_masks = False

        # Height and Width of Image after rescaling
        # Only Supporeted for `data_type` COCO
        self.image_height = 224
        self.image_width = 224
