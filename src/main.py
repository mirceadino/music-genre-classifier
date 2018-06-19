import argparse
import logging
import os

import tensorflow as tf

from config import config
from src.classifier.genre_mapper import GenreMapper
from src.classifier.music_genre_classifier import MusicGenreClassifier
from src.dataset.dataset import Dataset
from src.dataset.dataset_creator import DatasetCreator
from src.dataset.statistics import DatasetStatistics
from src.nn.neural_network import NeuralNetwork
from src.songs.utils import read_song_from_wav, download_yt_song

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


def refine_args(args):
    # Refine the mode of the program.
    if args.create:
        args.mode = "create_dataset"
    elif args.train > 0:
        args.mode = "train"
        args.epochs = args.train
        args.train = True
    elif len(args.test) > 0:
        args.mode = "test"
        args.test_datasets = args.test
        args.test = True
    elif args.predict:
        args.mode = "predict"

    # Refine dataset equalization.
    args.equalize = True
    if args.no_equalize:
        args.equalize = False

    print("Refined arguments for the program: {0}".format(args))

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


def train(nn, genre_mapper, dataset_creator, num_epochs):
    train_dataset = Dataset(name="training",
                            path=config.PATH_TRAINING_DATASET,
                            dataset_creator=dataset_creator)
    train_stats = DatasetStatistics(train_dataset, genre_mapper)
    train_stats.all()
    train_x, train_y = train_dataset.get()

    val_dataset = Dataset(name="validation",
                          path=config.PATH_VALIDATION_DATASET,
                          dataset_creator=dataset_creator)
    val_stats = DatasetStatistics(val_dataset, genre_mapper)
    val_stats.all()
    val_x, val_y = val_dataset.get()

    nn.train(train_x, train_y, val_x, val_y, num_epochs=num_epochs,
             batch_size=config.BATCH_SIZE, shuffle=config.SHUFFLE,
             snapshot_epoch=config.SNAPSHOT_EPOCH,
             snapshot_step=config.SNAPSHOT_STEP, show_metric=config.SHOW_METRIC)

    nn.save(config.PATH_MODEL)


def test(classifier, genre_mapper, dataset_creator, datasets):
    path = {"training": config.PATH_TRAINING_DATASET,
            "validation": config.PATH_VALIDATION_DATASET,
            "testing": config.PATH_TESTING_DATASET}

    for name in datasets:
        print("------")
        test_dataset = Dataset(name=name, path=path[name],
                               dataset_creator=dataset_creator)
        test_dataset.load(False)
        test_stats = DatasetStatistics(test_dataset, genre_mapper)
        test_stats.all()
        test_x, test_y = test_dataset.get()

        test_accuracy, genres_pred = classifier.test(test_x, test_y)
        print("Obtained accuracy for {0} dataset was: {1}.".format(
            test_dataset.name, test_accuracy))

        test_stats.confusion_matrix(genres_pred)


def predict(classifier, path):
    from_youtube = False
    if not os.path.isfile(path):
        from_youtube = True
        path = download_yt_song(path)

    song, rate = read_song_from_wav(path)
    print(classifier.predict_song_from_raw_song(song, rate))
    print(classifier.predict_counting_from_raw_song(song, rate))

    if from_youtube:
        os.remove(path)


def main():
    args = refine_args(parse_args())

    # Create entitites.
    genre_mapper = GenreMapper(config.GENRES)
    dataset_creator = DatasetCreator(genre_mapper,
                                     slice_height=config.SLICE_HEIGHT,
                                     slice_width=config.SLICE_WIDTH,
                                     slice_overlap=config.SLICE_OVERLAP)
    nn = None
    classifier = None
    if args.mode != "create_dataset":
        nn = NeuralNetwork(num_rows=config.SLICE_HEIGHT,
                           num_cols=config.SLICE_WIDTH,
                           num_classes=len(config.GENRES))
        classifier = MusicGenreClassifier(nn, genre_mapper, dataset_creator)
        if args.mode != "train" or args.resume:
            nn.load(config.PATH_MODEL)

    # Do actual work.
    if args.mode == "create_dataset":
        create_dataset(dataset_creator, args.equalize)

    elif args.mode == "train":
        train(nn, genre_mapper, dataset_creator, args.epochs)

    elif args.mode == "test":
        test(classifier, genre_mapper, dataset_creator, args.test_datasets)

    elif args.mode == "predict":
        predict(classifier, args.predict)


if __name__ == "__main__":
    main()
