import logging
import pickle
import random

import numpy as np

from src.songs.utils import read_songs_from_csv, read_song_from_wav, \
    song_to_spectrogram_slices


class DatasetCreator:
    def __init__(self, method_name, genre_mapper, x_shape, y_shape, **kwargs):
        self.__method = None
        self.__genre_mapper = genre_mapper
        self.__x_shape = x_shape
        self.__y_shape = y_shape
        self.__kwargs = kwargs

        if method_name == "raw_song_to_slices":
            self.__method = self.__raw_song_to_slices

        if self.__method is None:
            raise ValueError("Invalid method to convert raw song to data.")

    def create_dataset(self, path_raw_songs, path_raw_info, path_training,
                       path_validation, path_testing, ratio_validation,
                       ratio_testing):
        """Converts raw data to data that can be fed to the neural network.

        Methods:
            raw_song_to_slices: A song is divided into equally sized slices.
                All the resulting slices will be x with the song label as y.

                Method args:
                    slice_size (int): Size of the slice.
                    slice_overlap (int): Overlap of the slices.
                    genre_mapper (GenreMapper): Mapper to be used.

        Args:
            path_raw_songs (str): Path where the raw songs are stored.
            path_raw_info (str): Path where the raw song information is stored.
            path_training (str): Path where the training dataset will be stored.
            path_validation (str): Path where the cross-validation dataset will
                be stored.
            path_testing (str): Path where the testing dataset will be stored.
            ratio_validation (double): Ratio of the size of the cross-validation
                dataset. Must be between 0 and 1.
            ratio_testing (double): Ratio of the size of the testing dataset.
                Must be between 0 and 1.

        Returns:
            bool: True if successful. False otherwise.
        """

        method = self.__method
        kwargs = self.__kwargs

        logging.info("[+] Creating dataset...")

        all_songs = read_songs_from_csv(path_raw_info)
        dataset = []
        for song in all_songs:
            path = "".join([path_raw_songs, song.id, ".", song.audio_format])
            waveform, rate = read_song_from_wav(path)
            slices = method(waveform, rate, **kwargs)
            slices_and_labels = self.add_label(slices, song)
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

    def song_to_x(self, waveform, rate):
        return self.__method(waveform, rate, **self.__kwargs)

    def add_label(self, slices, song):
        label = self.__genre_mapper.genre_to_y(song.genre)
        return list(map(lambda x: (x, label), slices))

    def __raw_song_to_slices(self, waveform, rate, slice_size, slice_overlap):
        slices = song_to_spectrogram_slices(waveform, rate, slice_size,
                                            slice_overlap)
        return np.array(slices).reshape(self.__x_shape)
