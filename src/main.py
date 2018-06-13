import argparse
import logging
import os
import subprocess
import tensorflow as tf

from config import config
from src.classifier.genre_mapper import GenreMapper
from src.classifier.music_genre_classifier import MusicGenreClassifier
from src.dataset.dataset import Dataset
from src.dataset.dataset_creator import DatasetCreator
from src.dataset.statistics import DatasetStatistics
from src.nn.neural_network import NeuralNetwork
from src.songs.utils import read_song_from_wav

logging.getLogger().setLevel(logging.INFO)
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def parse_args():
    # TODO: Add documentation.
    parser = argparse.ArgumentParser(
        description="Music genre classifier tools.")

    mutually_exclusive = parser.add_mutually_exclusive_group()
    mutually_exclusive.add_argument("--create", action='store_true',
                                    help="Uses the program to create the "
                                         "dataset.")
    parser.add_argument("--no_equalize", action='store_true',
                        help="Used with --create. Whether to equalize or not "
                             "the dataset, which means to that all the "
                             "classes will have data of the same size.")

    mutually_exclusive.add_argument("--train", type=int, default=-1,
                                    help="Uses to program to perform training "
                                         "for the specified number of epochs.")
    parser.add_argument("--resume", action='store_true',
                        help="Used with --train. Load the model and continue "
                             "training on it.")

    mutually_exclusive.add_argument("--test", action='append', default=[],
                                    choices=["training", "validation",
                                             "testing"],
                                    help="Uses the program to test on the "
                                         "selected datasets.")

    mutually_exclusive.add_argument("--predict", type=str,
                                    help="Uses the program to predict. It is "
                                         "followed by the path to the song. "
                                         "If the path is a Youtube url ("
                                         "contains 'youtu' in the string), "
                                         "then it downloads the song")

    args = parser.parse_args()
    print("Given arguments for the program: {0}".format(args))
    return args


def process_args(args):
    # Refine the mode of the program.
    if args.create:
        args.mode = "create_dataset"
    elif args.train > 0:
        args.mode = "train"
    elif len(args.test) > 0:
        args.mode = "test"
    elif args.predict:
        args.mode = "predict"

    args.equalize = True
    if args.no_equalize:
        args.equalize = False

    return args


def create_dataset(creator, equalize):
    creator.create_dataset(path_raw_songs=config.PATH_SONGS,
                           path_raw_info=config.PATH_SONG_INFO,
                           path_training=config.PATH_TRAINING_DATASET,
                           path_validation=config.PATH_VALIDATION_DATASET,
                           path_testing=config.PATH_TESTING_DATASET,
                           ratio_validation=config.RATIO_VALIDATION,
                           ratio_testing=config.RATIO_TESTING,
                           equalize=equalize)


def train(nn, genre_mapper, num_epochs, x_shape, y_shape):
    logging.info("You're going to train the model on the existing dataset.")
    # TODO: Log information about the paths.

    train_dataset = Dataset(name="training",
                            path=config.PATH_TRAINING_DATASET,
                            x_shape=x_shape, y_shape=y_shape)
    train_stats = DatasetStatistics(train_dataset, genre_mapper)
    train_stats.all()
    train_x, train_y = train_dataset.get()

    val_dataset = Dataset(name="validation",
                          path=config.PATH_VALIDATION_DATASET,
                          x_shape=x_shape, y_shape=y_shape)
    val_stats = DatasetStatistics(val_dataset, genre_mapper)
    val_stats.all()
    val_x, val_y = val_dataset.get()

    nn.train(train_x, train_y, val_x, val_y, num_epochs=num_epochs,
             batch_size=config.BATCH_SIZE, shuffle=config.SHUFFLE,
             snapshot_epoch=config.SNAPSHOT_EPOCH,
             snapshot_step=config.SNAPSHOT_STEP, show_metric=config.SHOW_METRIC)

    nn.save(config.PATH_MODEL)


def test(nn, genre_mapper, datasets, x_shape, y_shape):
    path = {"training": config.PATH_TRAINING_DATASET,
            "validation": config.PATH_VALIDATION_DATASET,
            "testing": config.PATH_TESTING_DATASET}

    for name in datasets:
        print("------")
        test_dataset = Dataset(name=name, path=path[name],
                               x_shape=x_shape, y_shape=y_shape)
        test_stats = DatasetStatistics(test_dataset, genre_mapper)
        test_stats.all()
        test_x, test_y = test_dataset.get()

        test_accuracy = nn.test(test_x, test_y)
        print("Obtained accuracy for {0} dataset was: {1}.".format(
            test_dataset.name, test_accuracy))

        y_pred = []
        for i in range(0, len(test_x), config.BATCH_SIZE):
            j = min(len(test_x), i + config.BATCH_SIZE)
            x = test_x[i:j]
            label_pred = nn.predict_label(x)
            y_pred.extend(
                list(map(lambda labels: genre_mapper.label_to_y(labels[0]),
                         label_pred)))
        test_stats.confusion_matrix(y_pred)


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


def predict(nn, genre_mapper, dataset_creator, path):
    nn.load(config.PATH_MODEL)
    classifier = MusicGenreClassifier(nn, genre_mapper, dataset_creator)

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

    x_shape = [-1, config.SLICE_SIZE, config.SLICE_SIZE, 1]
    y_shape = [-1, len(config.GENRES)]
    genre_mapper = GenreMapper(config.GENRES)
    dataset_creator = DatasetCreator("raw_song_to_slices", genre_mapper,
                                     x_shape, y_shape,
                                     slice_size=config.SLICE_SIZE,
                                     slice_overlap=config.SLICE_OVERLAP)

    nn = None
    if mode != "create_dataset":
        nn = NeuralNetwork("cnn_for_slices", num_rows=config.SLICE_SIZE,
                           num_cols=config.SLICE_SIZE,
                           num_classes=len(config.GENRES))
        if mode != "train" or args.resume:
            nn.load(config.PATH_MODEL)

    if mode == "create_dataset":
        create_dataset(dataset_creator, args.equalize)

    elif mode == "train":
        train(nn, genre_mapper, args.train, x_shape, y_shape)

    elif mode == "test":
        test(nn, genre_mapper, args.test, x_shape, y_shape)

    elif mode == "predict":
        predict(nn, genre_mapper, dataset_creator, args.predict)


if __name__ == "__main__":
    main()
