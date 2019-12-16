# import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dropout, Activation
from tensorflow.keras.activations import relu, tanh
from tensorflow.keras.regularizers import l2


class ImageTranslationNetwork(Model):
    """
        Same as network in Luigis cycle_prior.

        Not supporting discriminator / Fully connected output.
        Support for this should be implemented as a separate class.
    """

    def __init__(
        self,
        input_chs,
        filter_spec,
        name,
        l2_lambda=1e-3,
        leaky_alpha=0.3,
        dropout_rate=0.2,
        dtype="float32",
    ):
        """
            Inputs:
                input_chs -         int, number of channels in input
                filter_spec -       list of integers specifying the filtercount
                                    for the respective layers
                name -              str, name of model
                leaky_alpha=0.3 -   float in [0,1], passed to the RELU
                                    activation of all but the last layer
                dropout_rate=0.2 -  float in [0,1], specifying the dropout
                                    probability in training time for all but
                                    the last layer
                dtype='float32' -   str or dtype, datatype of model
            Outputs:
                None
        """
        super(ImageTranslationNetwork, self).__init__(name=name, dtype=dtype)

        self.leaky_alpha = leaky_alpha
        self.dropout = Dropout(dropout_rate, dtype=dtype)

        conv_specs = {
            "kernel_size": 3,
            "strides": 1,
            "kernel_initializer": "GlorotNormal",
            "padding": "same",
            "kernel_regularizer": l2(l2_lambda),
            "bias_regularizer": l2(l2_lambda),
            "dtype": dtype,
        }
        layer = Conv2D(
            filter_spec[0],
            input_shape=(None, None, input_chs),
            name=f"{name}-{0:02d}",
            **conv_specs,
        )
        self.layers_ = [layer]
        for l, n_filters in enumerate(filter_spec[1:]):
            layer = Conv2D(n_filters, name=f"{name}-{l+1:02d}", **conv_specs)
            self.layers_.append(layer)

    def call(self, inputs, training=False):
        """ Implements the feed forward part of the network """
        x = inputs
        for layer in self.layers_[:-1]:
            x = layer(x)
            x = relu(x, alpha=self.leaky_alpha)
            x = self.dropout(x, training)
        x = self.layers_[-1](x)
        return tanh(x)
