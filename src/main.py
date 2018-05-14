import argparse
import logging
import os

import subprocess

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

    parser.add_argument("-r", "--resume", action='store_true',
                        help="In training mode, load the model and continue "
                             "training on it.")

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
                             "mode. If the path is a Youtube url (contains "
                             "'youtu' in the string), then it downloads the "
                             "song")

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


def train(load):
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
    if load:
        nn.load(config.PATH_MODEL)
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


def song_id(path):
    return path.replace("/", " ").replace("=", " ").split()[-1]


def get_youtube_url(path):
    return "https://www.youtube.com/watch?v=" + song_id(path)


def download_youtube_song(url):
    subprocess.call("youtube-dl "
                    "--extract-audio "
                    "--quiet "
                    "--audio-format {0} "
                    "--output {1} "
                    "{2}"
                    .format("wav", "%(id)s.%(ext)s", url).split())
    path = song_id(url) + ".wav"
    return path


def predict(path):
    nn = NeuralNetwork()
    nn.load(config.PATH_MODEL)
    classifier = MusicGenreClassifier(nn, config.GENRES)

    from_youtube = False
    if not os.path.isfile(path):
        from_youtube = True
        url = get_youtube_url(path)
        path = download_youtube_song(url)

    song, rate = read_song_from_wav(path)
    print(classifier.predict(song, rate))

    if from_youtube:
        subprocess.call("rm {0}".format(path).split())


def main():
    args = process_args(parse_args())
    mode = args.mode

    if mode == "create_dataset":
        create_dataset()

    elif mode == "train":
        train(args.resume)

    elif mode == "test":
        test(args.test)

    elif mode == "predict":
        path = args.path
        predict(path)


if __name__ == "__main__":
    main()
