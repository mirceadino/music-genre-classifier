import logging
import pickle
import random

import numpy as np

from src.songs.utils import read_songs_from_csv, read_song_from_wav, \
    song_to_spectrogram_slices


class DatasetCreator:
    def __init__(self, genre_mapper, x_shape, y_shape, slice_size,
                 slice_overlap):
        self.__genre_mapper = genre_mapper
        self.__x_shape = x_shape
        self.__y_shape = y_shape
        self.__slice_size = slice_size
        self.__slice_overlap = slice_overlap

    def create_dataset(self, path_raw_songs, path_raw_info, path_training,
                       path_validation, path_testing, ratio_validation,
                       ratio_testing, equalize):
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
            equalize (bool): Whether to equalize the dataset or not such that
                is the same number of samples for each label. This is done by
                resizing each category to the minimum positive number of
                samples.

        Returns:
            bool: True if successful. False otherwise.
        """

        logging.info("[+] Creating dataset...")

        all_songs = read_songs_from_csv(path_raw_info)
        dataset = []
        for song in all_songs:
            if song.genre not in self.__genre_mapper.genres:
                continue
            path = "".join([path_raw_songs, song.id, ".", song.audio_format])
            waveform, rate = read_song_from_wav(path)
            slices = self.song_to_slices(waveform, rate)
            label = self.__genre_mapper.genre_to_y(song.genre)
            dataset.append((slices, label))

        logging.info("[+] Dataset created!")

        logging.info("[+] Saving dataset...")

        if equalize:
            training_dataset, validation_dataset, testing_dataset = \
                self.__equalize(dataset, ratio_validation, ratio_testing)
        else:
            random.shuffle(dataset)
            i = int(len(dataset) * (1.0 - ratio_validation - ratio_testing))
            j = int(len(dataset) * (1.0 - ratio_testing))
            training_dataset = dataset[:i]
            validation_dataset = dataset[i:j]
            testing_dataset = dataset[j:]

        with open(path_training, "wb") as outfile:
            pickle.dump(training_dataset, outfile)
        logging.info("[+] {0} items have been saved to: {1}"
                     .format(len(training_dataset), path_training))

        with open(path_validation, "wb") as outfile:
            pickle.dump(validation_dataset, outfile)
        logging.info("[+] {0} items have been saved to: {1}"
                     .format(len(validation_dataset), path_validation))

        with open(path_testing, "wb") as outfile:
            pickle.dump(testing_dataset, outfile)
        logging.info("[+] {0} items have been saved to: {1}"
                     .format(len(testing_dataset), path_testing))

        logging.info("[+] Dataset saved!")

        return True

    def song_to_slices(self, waveform, rate):
        return song_to_spectrogram_slices(waveform, rate, self.__slice_size,
                                          self.__slice_overlap)

    def slices_to_x(self, slices):
        return np.array(slices).reshape(self.__x_shape)

    def song_to_x(self, waveform, rate):
        return self.slices_to_x(self.song_to_slices(waveform, rate))

    def add_label(self, slices, song):
        label = self.__genre_mapper.genre_to_y(song.genre)
        return list(map(lambda x: (x, label), slices))

    def __equalize(self, dataset, ratio_validation, ratio_testing):
        x_per_y = {}
        for x, y in dataset:
            if str(y) in x_per_y.keys():
                x_per_y[str(y)].append((x, y))
            else:
                x_per_y[str(y)] = [(x, y)]
        min_num_samples = len(dataset)
        for y in x_per_y.keys():
            min_num_samples = min(min_num_samples, len(x_per_y[y]))
        training_samples = []
        validation_samples = []
        testing_samples = []
        i = int(min_num_samples * (1.0 - ratio_validation - ratio_testing))
        j = int(min_num_samples * (1.0 - ratio_testing))
        for y in x_per_y.keys():
            random.shuffle(x_per_y[y])
            samples = x_per_y[y][:min_num_samples]
            training_samples.extend(samples[:i])
            validation_samples.extend(samples[i:j])
            testing_samples.extend(samples[j:])
        return training_samples, validation_samples, testing_samples
