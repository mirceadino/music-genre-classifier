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
    parser = argparse.ArgumentParser(
        description="Music genre classifier tools.")

    parser.add_argument("-m", "--mode", type=str,
                        choices=["create_dataset", "train", "test", "predict"],
                        help="Mode of the program. On testing mode, "
                             "by default, it tests only on the testing "
                             "dataset, unless specified otherwise by the test "
                             "parameter.")

    parser.add_argument("-mc", "--create", action='store_true',
                        help="Set mode to create_dataset. Doesn't override the "
                             "mode parameter.")

    parser.add_argument("-ml", "--train", action='store_true',
                        help="Set mode to train. Doesn't override the mode "
                             "parameter.")

    parser.add_argument("-mt", "--test", action='append', default=[],
                        choices=["training", "validation", "testing"],
                        help="Set mode to test and test on the named datasets. "
                             "Doesn't override the mode parameter.")

    parser.add_argument("-mp", "--predict", action='store_true',
                        help="Set mode to predict. Doesn't override the mode "
                             "parameter.")

    parser.add_argument("-p", "--path", type=str,
                        help="Path to the song. Only considered in predict "
                             "mode.")

    args = parser.parse_args()
    print("Given arguments for the program: {0}".format(args))
    return args


def process_args(args):
    # Refine the mode of the program.
    if args.mode is None:
        if args.create:
            args.mode = "create_dataset"
        elif args.train:
            args.mode = "train"
        elif len(args.test) > 0:
            args.mode = "test"
        elif args.predict:
            args.mode = "predict"

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
    train_dataset.display_statistics(all_stats=True)
    train_x, train_y = train_dataset.get()

    val_dataset = Dataset(name="validation",
                          path=config.PATH_VALIDATION_DATASET)
    val_dataset.display_statistics(all_stats=True)
    val_x, val_y = val_dataset.get()

    nn = NeuralNetwork()
    nn.train(train_x, train_y, val_x, val_y)

    nn.save(config.PATH_MODEL)


def test(datasets):
    nn = NeuralNetwork()
    nn.load(config.PATH_MODEL)

    path = {"training": config.PATH_TRAINING_DATASET,
            "validation": config.PATH_VALIDATION_DATASET,
            "testing": config.PATH_TESTING_DATASET}

    for name in datasets:
        test_dataset = Dataset(name=name, path=path[name])
        test_dataset.display_statistics(all_stats=True)
        test_x, test_y = test_dataset.get()

        test_accuracy = nn.test(test_x, test_y)
        print("Obtained accuracy for {0} dataset was: {1}.".format(
            test_dataset.name, test_accuracy))


def predict(path):
    nn = NeuralNetwork()
    nn.load(config.PATH_MODEL)
    classifier = MusicGenreClassifier(nn, config.GENRES)
    song, rate = read_song_from_wav(path)
    print(classifier.predict(song, rate))


def main():
    args = process_args(parse_args())
    mode = args.mode

    if mode == "create_dataset":
        create_dataset()

    elif mode == "train":
        train()

    elif mode == "test":
        test(args.test)

    elif mode == "predict":
        path = args.path
        predict(path)


if __name__ == "__main__":
    main()
