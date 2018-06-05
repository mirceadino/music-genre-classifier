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


def parse_args():
    # TODO: Add documentation.
    parser = argparse.ArgumentParser(
        description="Music genre classifier tools.")

    parser.add_argument("--create", action='store_true',
                        help="Uses the program to create the dataset.")

    parser.add_argument("--train", type=int, default=-1,
                        help="Uses to program to perform training for the "
                             "specified number of epochs.")

    parser.add_argument("--resume", action='store_true',
                        help="In training mode, load the model and continue "
                             "training on it.")

    parser.add_argument("--test", action='append', default=[],
                        choices=["training", "validation", "testing"],
                        help="Uses the program to test on the selected "
                             "datasets.")

    parser.add_argument("--predict", type=str,
                        help="Uses the program to predict. It is followed by "
                             "the path to the song. Path to the song. If the "
                             "path is a Youtube url (contains 'youtu' in the "
                             "string), then it downloads the song")

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

    return args


def create_dataset(genre_mapper, creator):
    creator.create_dataset(path_raw_songs=config.PATH_SONGS,
                           path_raw_info=config.PATH_SONG_INFO,
                           path_training=config.PATH_TRAINING_DATASET,
                           path_validation=config.PATH_VALIDATION_DATASET,
                           path_testing=config.PATH_TESTING_DATASET,
                           ratio_validation=config.RATIO_VALIDATION,
                           ratio_testing=config.RATIO_TESTING, equalize=True)


def train(num_epochs, load, genre_mapper, x_shape, y_shape):
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

    nn = NeuralNetwork("cnn_for_slices", num_rows=config.SLICE_SIZE,
                       num_cols=config.SLICE_SIZE,
                       num_classes=len(config.GENRES))
    if load:
        nn.load(config.PATH_MODEL)
    nn.train(train_x, train_y, val_x, val_y, num_epochs=num_epochs,
             batch_size=config.BATCH_SIZE, shuffle=config.SHUFFLE,
             snapshot_epoch=config.SNAPSHOT_EPOCH,
             snapshot_step=config.SNAPSHOT_STEP, show_metric=config.SHOW_METRIC)

    nn.save(config.PATH_MODEL)


def test(datasets, genre_mapper, x_shape, y_shape):
    nn = NeuralNetwork("cnn_for_slices", num_rows=config.SLICE_SIZE,
                       num_cols=config.SLICE_SIZE,
                       num_classes=len(config.GENRES))
    nn.load(config.PATH_MODEL)

    path = {"training": config.PATH_TRAINING_DATASET,
            "validation": config.PATH_VALIDATION_DATASET,
            "testing": config.PATH_TESTING_DATASET}

    for name in datasets:
        test_dataset = Dataset(name=name, path=path[name],
                               x_shape=x_shape, y_shape=y_shape)
        test_stats = DatasetStatistics(test_dataset, genre_mapper)
        test_stats.all()
        test_x, test_y = test_dataset.get()

        test_accuracy = nn.test(test_x, test_y)
        print("Obtained accuracy for {0} dataset was: {1}.".format(
            test_dataset.name, test_accuracy))

        label_pred = nn.predict_label(test_x)
        y_pred = list(map(lambda labels: genre_mapper.label_to_y(labels[0]),
                          label_pred))
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


def predict(path, genre_mapper, dataset_creator):
    nn = NeuralNetwork("cnn_for_slices", num_rows=config.SLICE_SIZE,
                       num_cols=config.SLICE_SIZE,
                       num_classes=len(config.GENRES))
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

    if mode == "create_dataset":
        create_dataset(genre_mapper, dataset_creator)

    elif mode == "train":
        train(args.train, args.resume, genre_mapper, x_shape, y_shape)

    elif mode == "test":
        test(args.test, genre_mapper, x_shape, y_shape)

    elif mode == "predict":
        predict(args.predict, genre_mapper, dataset_creator)


if __name__ == "__main__":
    main()
