import argparse
import logging

from config import config
from src.classifier.music_genre_classifier import MusicGenreClassifier
from src.classifier.nn.neural_network import NeuralNetwork
from src.dataset.dataset import Dataset
from src.dataset.dataset_creator import DatasetCreator
from src.utils.song_utils import read_song_from_wav

logging.getLogger().setLevel(logging.INFO)


def parse_args():
    # TODO: Add documentation.
    parser = argparse.ArgumentParser(description="Music genre classifier tools.")
    parser.add_argument("-m", "--mode", type=str,
                        help="<Required> Mode of the program",
                        choices=["create_dataset", "train", "test", "predict"],
                        required=True)
    parser.add_argument("-p", "--path", type=str,
                        help="Path to the song. Only considered in predict mode.")
    args = parser.parse_args()
    print("Given arguments for the program: {0}".format(args))
    return args


def create_dataset():
    creator = DatasetCreator()
    creator.create(path_raw_songs=config.PATH_SONGS,
                   path_raw_info=config.PATH_SONG_INFO,
                   path_training=config.PATH_TRAINING_DATASET,
                   path_validation=config.PATH_VALIDATION_DATASET,
                   path_testing=config.PATH_TESTING_DATASET,
                   ratio_validation=config.RATIO_VALIDATION,
                   ratio_testing=config.RATIO_TESTING)


def train():
    logging.info("You're going to train the model on the existing dataset.")
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


def test():
    test_dataset = Dataset(name="testing", path=config.PATH_TESTING_DATASET)
    test_x, test_y = test_dataset.get()

    nn = NeuralNetwork()
    nn.load(config.PATH_MODEL)

    accuracy = nn.test(test_x, test_y)
    print("Obtained accuracy was: {0}.".format(accuracy))


def predict(path):
    nn = NeuralNetwork()
    nn.load(config.PATH_MODEL)
    classifier = MusicGenreClassifier(nn, config.GENRES)
    song, rate = read_song_from_wav(path)
    print(classifier.predict(song))


def main():
    args = parse_args()
    mode = args.mode

    if mode == "create_dataset":
        create_dataset()

    elif mode == "train":
        train()

    elif mode == "test":
        test()

    elif mode == "predict":
        path = args.path
        predict(path)


if __name__ == "__main__":
    main()
