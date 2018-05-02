import logging

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression

from config import config


class NeuralNetwork:
    """Neural network that allows loading, saving, training, testing and
    predicting."""

    def __init__(self):
        self.__model = self.__create_model(config.NUM_CLASSES,
                                           config.SLICE_SIZE)

    def load(self, path):
        """Loads model from file.

        Args:
            path (str): Path to the file where the model is stored.
        """
        logging.info("[+] Loading model...")
        self.__model.load(path)
        logging.info("[+] Loading finished!")

    def save(self, path):
        """Saves model to a file.

        Args:
            path (str): Path to the file where the model will be stored.
        """
        logging.info("[+] Saving model...")
        self.__model.save(path)
        logging.info("[+] Saving finished!")

    def train(self, train_x, train_y, val_x, val_y):
        """Trains the model based on the given datasets.

        Args:
            train_x (numpy.array): Training input.
            train_y (numpy.array): Correct outputs for training.
            val_x (numpy.array): Validation input.
            val_y (numpy.array): Correct outputs for validation.
        """
        logging.info("[+] Starting testing...")
        self.__model.fit(train_x, train_y, n_epoch=config.NUM_EPOCHS,
                         batch_size=config.BATCH_SIZE,
                         shuffle=config.SHUFFLE, validation_set=(val_x, val_y),
                         snapshot_epoch=config.SNAPSHOT_EPOCH,
                         snapshot_step=config.SNAPSHOT_STEP,
                         show_metric=config.SHOW_METRIC)
        logging.info("[+] Training finished!")

    def test(self, test_x, test_y):
        """Tests the model on the given dataset.

        Args:
            test_x (numpy.array): Testing input.
            test_y (numpy.array): Correct outputs.

        Returns:
            float: Top-1 accuracy of the model on the testing dataset.
        """
        logging.info("[+] Starting testing...")
        accuracy = self.__model.evaluate(test_x, test_y)[0]
        logging.info("[+] Testing finished!")
        return accuracy

    def predict_label(self, x):
        """Predicts the labels of x in decreasing order of likelihood.

        Args:
            x (array or list of array): Input for which we want the prediction.

        Returns:
            array or list of array: Predicted labels in decreasing order of
            likelihood.

        Example:
            If the predicted probabilities are [0.2 0.5 0.3], the method will
            return [1 2 0]. The first element is the label with the highest
            likelihood to be correct.
        """
        return self.__model.predict_label(x)

    def predict_probabilities(self, x):
        """Predicts the likelihood of x to each class.

        Args:
            x (array or list of array): Input for which we want the prediction.

        Returns:
            array or list of array: Predicted probabilities.
        """
        return self.__model.predict(x)

    def __create_model(self, num_classes, slice_size):
        """Creates a model. The model will have:
        - an input layer of shape (batch_size, 128, slice_size, 1)
        - convolutional layers
        - fully connected layers
        - an output regression layer of shape (batch_size, num_classes)

        Args:
            num_classes (int): Number of classes (for the output layer).
            slice_size (int): Size of the slice (for the input layer).

        Returns:
            tflearn.models.dnn.DNN: Model.
        """
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
        #network = dropout(network, 0.5)

        network = fully_connected(network, 1024, activation='elu')
        #network = dropout(network, 0.5)

        network = fully_connected(network, num_classes, activation='softmax')
        network = regression(network, optimizer='sgd', loss='categorical_crossentropy')

        model = tflearn.DNN(network)

        logging.info("[+] Model created!")
        return model
