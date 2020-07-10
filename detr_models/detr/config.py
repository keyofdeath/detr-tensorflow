class DefaultDETRConfig:
    def __init__(self):
        # Data Storage

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
        self.batch_size = 2
        self.learning_rate = 1e-4

        # Backbone
        self.backbone_name = "ResNet50"
