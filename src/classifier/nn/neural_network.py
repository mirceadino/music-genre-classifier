import logging

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

from config import config


class NeuralNetwork:
    # TODO: Add documentation for this class.

    def __init__(self):
        self.__model = self.__create_model(config.NUM_CLASSES, config.SLICE_SIZE)

    def load(self, path):
        # TODO: Add documentation for this method.
        logging.info("[+] Loading model...")
        self.__model.load(path)
        logging.info("[+] Loading finished!")

    def save(self, path):
        # TODO: Add documentation for this method.
        logging.info("[+] Saving model...")
        self.__model.save(path)
        logging.info("[+] Saving finished!")

    def train(self, train_x, train_y, val_x, val_y):
        # TODO: Add documentation for this method.
        logging.info("[+] Starting testing...")
        self.__model.fit(train_x, train_y, n_epoch=config.NUM_EPOCHS,
                         batch_size=config.BATCH_SIZE, shuffle=config.SHUFFLE,
                         validation_set=(val_x, val_y), snapshot_epoch=config.SNAPSHOT_EPOCH,
                         snapshot_step=config.SNAPSHOT_STEP, show_metric=config.SHOW_METRIC)
        logging.info("[+] Training finished!")

    def test(self, test_x, test_y):
        # TODO: Add documentation for this method.
        logging.info("[+] Starting testing...")
        accuracy = self.__model.evaluate(test_x, test_y)
        logging.info("[+] Testing finished!")
        return accuracy

    def predict_label(self, x):
        # TODO: Add documentation for this method.
        # Note: x is list
        return self.__model.predict_label(x)

    def predict_probabilities(self, x):
        # TODO: Add documentation for this method.
        # Note: x is list
        return self.__model.predict(x)

    def __create_model(self, num_classes, slice_size):
        # TODO: Add documentation for this class.
        logging.info("[+] Creating model...")

        network = input_data(shape=[None, 128, slice_size, 1], name='input')

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

        network = fully_connected(network, 1024, activation='elu')
        network = dropout(network, 0.5)

        network = fully_connected(network, num_classes, activation='softmax')
        network = regression(network, optimizer='rmsprop', loss='categorical_crossentropy')

        model = tflearn.DNN(network)

        logging.info("[+] Model created!")
        return model
