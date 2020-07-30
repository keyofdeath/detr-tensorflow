class DefaultDETRConfig:
    def __init__(self):

        # Number of Predictions per Image
        self.num_queries = 100

        # Number of Classes
        self.num_classes = 4

        # Transformer Config
        self.dim_transformer = 256
        self.dim_feedforward = 2048
        self.num_transformer_layer = 6
        self.num_heads = 8

        # Training Config
        self.epochs = 300
        self.batch_size = 2
        self.learning_rate = 1e-4
        self.drops = [100, 200]
        self.weight_decay = 1e-4

        # Backbone Config
        self.backbone_name = "ResNet50"
