import logging

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

from config import config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class NeuralNetwork:
    # TODO: Add documentation for this class.

    def __init__(self):
        self.__model = self.__create_model(config.NUM_CLASSES, config.IMAGE_SIZE)

    def load(self, path):
        # TODO: Add documentation for this method.
        self.__model.load(path)

    def save(self, path):
        # TODO: Add documentation for this method.
        self.__model.save(path)

    def train(self, train_x, train_y, val_x, val_y):
        # TODO: Add documentation for this method.
        self.__model.fit(train_x, train_y, n_epoch=config.NUM_EPOCHES,
                         batch_size=config.BATCH_SIZE, shuffle=config.SHUFFLE,
                         validation_set=(val_x, val_y), snapshot_epoch=config.SNAPSHOT_EPOCH,
                         snapshot_step=config.SNAPSHOT_STEP, show_metric=config.SHOW_METRIC)

    def test(self, test_x, test_y):
        # TODO: Add documentation for this method.
        return self.__model.evaluate(test_x, test_y)

    def predict_label(self, x):
        # TODO: Add documentation for this method.
        # Note: x is list
        return self.__model.predict_label(x)

    def __create_model(self, num_classes, image_size):
        # TODO: Add documentation for this class.
        logger.info("[+] Creating model...")

        network = input_data(shape=[None, image_size, image_size, 1], name='input')

        network = conv_2d(network, 64, 2, activation='elu', weights_init="Xavier")
        network = max_pool_2d(network, 2)

        network = conv_2d(network, 128, 2, activation='elu', weights_init="Xavier")
        network = max_pool_2d(network, 2)

        network = conv_2d(network, 256, 2, activation='elu', weights_init="Xavier")
        network = max_pool_2d(network, 2)

        network = conv_2d(network, 512, 2, activation='elu', weights_init="Xavier")
        network = max_pool_2d(network, 2)

        network = fully_connected(network, 1024, activation='elu')
        network = dropout(network, 0.5)

        network = fully_connected(network, num_classes, activation='softmax')
        network = regression(network, optimizer='rmsprop', loss='categorical_crossentropy')

        model = tflearn.DNN(network)

        logger.info("[+] Model created!")
        return model
