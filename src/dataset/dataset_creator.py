import logging
import pickle
import random

from config import config
from src.utils.song_utils import read_songs_from_csv, read_song_from_wav, \
    song_to_spectrogram_slices


class DatasetCreator:
    """Converts raw data to data that can be used for training, validating and
    testing the learning model.
    """

    def __init__(self):
        """Creates a dataset creator."""
        pass

    def create(self, path_raw_songs, path_raw_info, path_training,
               path_validation, path_testing, ratio_validation=0,
               ratio_testing=0):
        """Converts raw data to data that can be fed to the classifier.

        Args:
            path_raw_songs (str): Path where the raw songs are stored.
            path_raw_info (str): Path where the raw song information is stored.
            path_training (str): Path where the training dataset will be stored.
            path_validation (str): Path where the cross-validation dataset will
                be stored.
            path_testing (str): Path where the testing dataset will be stored.
            ratio_validation (double): Ratio of the size of the cross-validation
                dataset. Must be between 0 and 1. Defaults is 0.
            ratio_testing (double): Ratio of the size of the testing dataset.
                Must be between 0 and 1. Defaults is 0.

        Returns:
            bool: True if successful. False otherwise.
        """
        logging.info("[+] Creating dataset...")

        all_songs = read_songs_from_csv(path_raw_info)
        dataset = []
        for song in all_songs:
            path = "".join([path_raw_songs, song.id, ".", song.audio_format])
            waveform, rate = read_song_from_wav(path)
            slices = song_to_spectrogram_slices(waveform, rate,
                                                config.SLICE_SIZE,
                                                config.SLICE_OVERLAP)
            slices_and_labels = list(map(lambda x: (x, song.get_y()), slices))
            dataset.extend(slices_and_labels)

        logging.info("[+] Dataset created!")

        logging.info("[+] Saving dataset...")

        random.shuffle(dataset)
        i = int(len(dataset) * (1.0 - ratio_validation - ratio_testing))
        j = int(len(dataset) * (1.0 - ratio_testing))
        training_dataset = dataset[:i]
        validation_dataset = dataset[i:j]
        testing_dataset = dataset[j:]

        with open(path_training, "wb") as outfile:
            pickle.dump(training_dataset, outfile)
        logging.info("[+] {0} slices have been saved to: {1}"
                     .format(len(training_dataset), path_training))

        with open(path_validation, "wb") as outfile:
            pickle.dump(validation_dataset, outfile)
        logging.info("[+] {0} slices have been saved to: {1}"
                     .format(len(validation_dataset), path_validation))

        with open(path_testing, "wb") as outfile:
            pickle.dump(testing_dataset, outfile)
        logging.info("[+] {0} slices have been saved to: {1}"
                     .format(len(testing_dataset), path_testing))

        logging.info("[+] Dataset saved!")

        return True
