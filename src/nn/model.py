import logging

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, fully_connected, dropout
from tflearn.layers.estimator import regression


class ModelFactory:
    # TODO: Add documentation about the class.

    @staticmethod
    def get(name, **kwargs):
        # TODO: Add documentation about the method.
        model = None

        logging.info("[+] Creating model...")

        if name is "cnn_for_slices":
            model = ModelFactory.__cnn_for_slices(**kwargs)

        if model is None:
            raise ValueError("Model \"{0}\" not found.".format(name))

        logging.info("[+] Model created!")

        return model

    @staticmethod
    def __cnn_for_slices(num_rows, num_cols, num_classes):
        """The model will have:
        - an input layer of shape (batch_size, num_rows, num_cols, 1)
        - convolutional layers (conv_2d and max_pool_2d)
        - fully connected and dropout layers
        - an output regression layer of shape (batch_size, num_classes)

        Args:
            num_rows (int): Number of rows in a slice (for the input layer).
            num_cols (int): Number of cols in a slice (for the input layer).
            num_classes (int): Number of classes (for the output layer).

        Returns:
            tflearn.models.dnn.DNN: Model.
        """
        network = input_data(shape=[None, num_rows, num_cols, 1], name='input')

        network = conv_2d(network, nb_filter=64, filter_size=2,
                          activation='elu', weights_init="Xavier")
        network = max_pool_2d(network, kernel_size=2)

        network = conv_2d(network, nb_filter=128, filter_size=2,
                          activation='elu', weights_init="Xavier")
        network = max_pool_2d(network, kernel_size=2)

        network = conv_2d(network, nb_filter=256, filter_size=2,
                          activation='elu', weights_init="Xavier")
        network = max_pool_2d(network, kernel_size=2)

        network = conv_2d(network, nb_filter=512, filter_size=2,
                          activation='elu', weights_init="Xavier")
        network = max_pool_2d(network, kernel_size=2)

        network = fully_connected(network, n_units=1024, activation='elu')
        network = dropout(network, keep_prob=0.5)

        network = fully_connected(network, n_units=num_classes,
                                  activation='softmax')
        network = regression(network, optimizer='rmsprop',
                             loss='categorical_crossentropy')

        model = tflearn.DNN(network)

        return model