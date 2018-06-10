import logging

from src.nn.model import ModelFactory


class NeuralNetwork:
    """Neural network that allows loading, saving, training, testing and
    predicting."""

    def __init__(self, model_name, **kwargs):
        self.__model = ModelFactory.get(model_name, **kwargs)

    def load(self, path):
        """Loads model from file.

        Args:
            path (str): Path to the file where the model is stored.
        """
        logging.info("[+] Loading model...")
        self.__model.load(path, weights_only=True)
        logging.info("[+] Loading finished!")

    def save(self, path):
        """Saves model to a file.

        Args:
            path (str): Path to the file where the model will be stored.
        """
        logging.info("[+] Saving model...")
        self.__model.save(path)
        logging.info("[+] Saving finished!")

    def train(self, train_x, train_y, val_x, val_y, num_epochs, batch_size,
              shuffle, snapshot_epoch, snapshot_step, show_metric):
        """Trains the model based on the given datasets.

        Args:
            train_x (numpy.array): Training input.
            train_y (numpy.array): Correct outputs for training.
            val_x (numpy.array): Validation input.
            val_y (numpy.array): Correct outputs for validation.
            num_epochs (int): Number of epochs for training.
            batch_size (int): Size of the batch for training.
            shuffle (bool): Whether to shuffle the dataset or not.
            snapshot_epoch (bool): Whether to make a snapshot after each
                epoch or not.
            snapshot_step (int or None): If int, snapshot the model after
                every snapshot_step steps.
            show_metric (bool): Whether to display accuracy or not.
        """
        logging.info("[+] Starting testing...")
        self.__model.fit(train_x, train_y, n_epoch=num_epochs,
                         batch_size=batch_size, shuffle=shuffle,
                         validation_set=(val_x, val_y),
                         snapshot_epoch=snapshot_epoch,
                         snapshot_step=snapshot_step, show_metric=show_metric)
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
