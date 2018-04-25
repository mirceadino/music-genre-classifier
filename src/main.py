import logging

from config import config
from src.classifier.music_genre_classifier import MusicGenreClassifier
from src.classifier.nn.neural_network import NeuralNetwork
from src.dataset.dataset import Dataset
from src.dataset.dataset_creator import DatasetCreator

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main():
    mode = "create_dataset"
    if mode == "create_dataset":
        creator = DatasetCreator()
        creator.create(path_raw=config.PATH_SONGS,
                       path_training=config.PATH_TRAINING_DATASET,
                       path_validation=config.PATH_VALIDATION_DATASET,
                       path_testing=config.PATH_TESTING_DATASET,
                       ratio_validation=config.RATIO_VALIDATION,
                       ratio_testing=config.RATIO_TESTING)

    elif mode == "train":
        logger.info("You're going to train the model on the existing dataset.")
        # TODO: Log information about the paths.

        train_dataset = Dataset(name="training",
                                path=config.PATH_TRAINING_DATASET)
        train_x, train_y = train_dataset.get()

        val_dataset = Dataset(name="validation",
                              path=config.PATH_VALIDATION_DATASET)
        val_x, val_y = val_dataset.get()

        nn = NeuralNetwork()
        nn.train(train_x, train_y, val_x, val_y)

        nn.save(config.PATH_MODEL)

    elif mode == "test":
        test_dataset = Dataset(name="testing", path=config.PATH_TESTING_DATASET)
        test_x, test_y = test_dataset.get()

        nn = NeuralNetwork()
        nn.load(config.PATH_MODEL)

        accuracy = nn.test(test_x, test_y)
        print("Obtained accuracy was: {0}.".format(accuracy))

    elif mode == "predict":
        nn = NeuralNetwork()
        nn.load(config.PATH_MODEL)
        classifier = MusicGenreClassifier(nn, config.GENRES)


if __name__ == "__main__":
    main()