# IPDB can be used for debugging.
# Ignoring flake8 error code F401
import ipdb  # noqa: F401
import tensorflow as tf
import tensorflow_addons as tfa


class SegmentationHead(tf.keras.Model):
    """Segmentation Head for Mask prediction"""

    def __init__(self, num_heads: int, dim_transformer: int, **kwargs):
        """Initialize SegmentationHead.

        Parameters
        ----------
        num_heads : int
            Number of heads in multi-head attention layers.
        dim_transformer : int
            Number of neurons in multi-head attention layers.
            Should be a multiple of `num_heads`.
        """
        super(SegmentationHead, self).__init__(**kwargs)

        dims_inter = [
            num_heads + dim_transformer,
            dim_transformer // 2,
            dim_transformer // 4,
            dim_transformer // 8,
            dim_transformer // 16,
        ]

        self.conv1 = tf.keras.layers.Conv2D(
            filters=dims_inter[0], kernel_size=3, padding="same"
        )
        self.gnorm1 = tfa.layers.GroupNormalization(groups=8, axis=3)

        self.conv2 = tf.keras.layers.Conv2D(
            filters=dims_inter[1], kernel_size=3, padding="same"
        )
        self.gnorm2 = tfa.layers.GroupNormalization(groups=8, axis=3)

        self.conv3 = tf.keras.layers.Conv2D(
            filters=dims_inter[2], kernel_size=3, padding="same"
        )
        self.gnorm3 = tfa.layers.GroupNormalization(groups=8, axis=3)

        self.conv4 = tf.keras.layers.Conv2D(
            filters=dims_inter[3], kernel_size=3, padding="same"
        )
        self.gnorm4 = tfa.layers.GroupNormalization(groups=8, axis=3)

        self.conv5 = tf.keras.layers.Conv2D(
            filters=dims_inter[4], kernel_size=3, padding="same"
        )
        self.gnorm5 = tfa.layers.GroupNormalization(groups=8, axis=3)

        self.conv_out = tf.keras.layers.Conv2D(
            filters=1, kernel_size=3, activation="sigmoid", padding="same"
        )

        self.adapter1 = tf.keras.layers.Conv2D(filters=dims_inter[1], kernel_size=1)
        self.adapter2 = tf.keras.layers.Conv2D(filters=dims_inter[2], kernel_size=1)
        self.adapter3 = tf.keras.layers.Conv2D(filters=dims_inter[3], kernel_size=1)

    def call(self, inputs):
        """Forward pass of SegmentationHead.

        Parameters
        ----------
        transformer_input : tf.Tensor
            Reduced feature map of shape (Batch Size, H_4, W_4, d), where `H_4` and
            `W_4` are the shapes of the last (in our case fourth) ResNet50 layer.
        attention_hmaps : tf.Tensor
            Number of heads attntion heatmaps for each object of shape (Batch Size, #Queries, #Heads, H, W),
            where ´H` and `W` are the shapes of the last ResNet50 layer.
        fpn_maps : List[tf.Tensor]
            Intermediate feature maps of ResNet50 layer, each of shape (Batch Size, H_x, W_x, D_x),
            where `H_x`, `W_x` and `D_x` are the corresponding height, width and number of dimensions.

        Returns
        -------
        x : tf.Tensor
            Mask predictions of shape (Batch Size, #Queries, H_1, W_1), where
            `H_1` and `W_1` are the shapes of the first feature map coming from
            the ResNet50 network.
        """
        transformer_input, attention_hmaps, fpn_maps = inputs

        _, _, nh, h, w = attention_hmaps.shape
        bs = tf.shape(attention_hmaps)[0]
        nq = tf.shape(attention_hmaps)[1]

        attention_hmaps = tf.reshape(attention_hmaps, shape=(bs * nq, h, w, nh))

        transformer_input = tf.repeat(transformer_input, nq, axis=0)

        x = tf.concat([attention_hmaps, transformer_input], axis=-1)

        # First Conv. Layer
        x = self.conv1(x)
        x = self.gnorm1(x)
        x = tf.keras.activations.relu(x)

        # Second Conv. Layer
        x = self.conv2(x)
        x = self.gnorm2(x)
        x = tf.keras.activations.relu(x)

        # First Adapter
        cur_fpn = self.adapter1(fpn_maps[2])
        cur_fpn = tf.repeat(cur_fpn, nq, axis=0)
        x = tf.image.resize(x, size=cur_fpn.shape[1:3], method="nearest")
        x = x + cur_fpn

        # # Third Conv. Layer
        x = self.conv3(x)
        x = self.gnorm3(x)
        x = tf.keras.activations.relu(x)

        # #Second Adapter
        cur_fpn = self.adapter2(fpn_maps[1])
        cur_fpn = tf.repeat(cur_fpn, nq, axis=0)
        x = tf.image.resize(x, size=cur_fpn.shape[1:3], method="nearest")
        x = x + cur_fpn

        # # Fourth Conv. Layer
        x = self.conv4(x)
        x = self.gnorm4(x)
        x = tf.keras.activations.relu(x)

        # # Third Adapter
        cur_fpn = self.adapter3(fpn_maps[0])
        cur_fpn = tf.repeat(cur_fpn, nq, axis=0)
        x = tf.image.resize(x, size=cur_fpn.shape[1:3], method="nearest")
        x = x + cur_fpn

        # # Fourth Conv. Layer
        x = self.conv5(x)
        x = self.gnorm5(x)
        x = tf.keras.activations.relu(x)

        x = self.conv_out(x)

        x = tf.reshape(
            tf.squeeze(x, axis=-1), shape=(bs, nq, tf.shape(x)[1], tf.shape(x)[2])
        )
        return x
