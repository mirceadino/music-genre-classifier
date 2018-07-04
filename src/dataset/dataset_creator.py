import logging
import random

import numpy as np

from src.songs.utils import read_songs_from_csv, read_song_from_wav, \
    song_to_spectrogram_slices, dump_to_file


class DatasetCreator:
    """Creates datasets (training, validation, testing) from raw input (reads
    metadata and .wav files), provides a method to convert an in-memory
    raw song to a form to be fed to the model as input, and provides methods
    to reshape data for the model."""

    def __init__(self, mapper, slice_height, slice_width, slice_overlap):
        """Creates a creator.
        The shape of x will be set to [-1, slice_height, slice_width, 1].
        The shape of y will be set to [-1, num_genres].

        Args:
            mapper (classifier.GenreMapper): Mapper for genres.
            slice_height (int): Height of slice for reshaping (usually 128
            because that's what the mel-spectrogram does, so it should be set
            automatically somehow).
            slice_width (int): Width of a slice for slicing up and reshaping.
            slice_overlap (int): Overlap between slices for slicing up. A
            negative overlap indicates space between slices.
        """
        self.__mapper = mapper
        self.__x_shape = [-1, slice_height, slice_width, 1]
        self.__y_shape = [-1, len(mapper.genres)]
        self.__slice_width = slice_width
        self.__slice_overlap = slice_overlap

    def create_dataset(self, path_raw_songs, path_raw_info, path_training,
                       path_validation, path_testing, ratio_validation,
                       ratio_testing, equalize):
        """Converts raw data to data that can be fed to the neural network.

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
                samples. If set to True, the datasets will have (
                usually) uniform distribution.
        """

        logging.info("[+] Creating dataset...")

        all_songs = read_songs_from_csv(path_raw_info)
        dataset = []
        for song in all_songs:
            if song.genre not in self.__mapper.genres:
                continue
            path = "".join([path_raw_songs, song.id, ".", song.audio_format])
            waveform, rate = read_song_from_wav(path)
            slices = self.song_to_slices(waveform, rate)
            label = self.__mapper.genre_to_y(song.genre)
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

        dump_to_file(training_dataset, path_training)
        logging.info("[+] {0} items have been saved to: {1}"
                     .format(len(training_dataset), path_training))

        dump_to_file(validation_dataset, path_validation)
        logging.info("[+] {0} items have been saved to: {1}"
                     .format(len(validation_dataset), path_validation))

        dump_to_file(testing_dataset, path_testing)
        logging.info("[+] {0} items have been saved to: {1}"
                     .format(len(testing_dataset), path_testing))

        logging.info("[+] Dataset saved!")

    def song_to_slices(self, waveform, rate):
        """Converts a raw song to a list of slices to be fed to the model.

        Args:
            waveform (list): Samples of the song.
            rate (int): Sampling rate of the song.

        Returns:
            slices (list of numpy.array's): List of slices.
        """
        return song_to_spectrogram_slices(waveform, rate, self.__slice_width,
                                          self.__slice_overlap)

    def slices_to_x(self, slices):
        """Reshapes the slices to the shape accepted by the model.

        Args:
            slices (list or numpy.array): Slices.

        Return:
            new_x (numpy.array): Reshaped output.
        """
        return np.array(slices).reshape(self.__x_shape)

    def song_to_x(self, waveform, rate):
        """Converts a song to the vectorial form accepted by the model.

        Args:
            waveform (list): Samples of the song.
            rate (int): Sampling rate of the song.

        Return:
            new_x (numpy.array): Reshaped output.
        """
        return self.slices_to_x(self.song_to_slices(waveform, rate))

    def reshape_x(self, x):
        """Reshapes a numpy.array to the shape accepted by the model.

        Args:
            x (list or numpy.array): Input.

        Return:
            new_x (numpy.array): Reshaped output.
        """
        return np.array(list(x)).reshape(self.__x_shape)

    def reshape_y(self, y):
        """Reshapes a numpy.array to the shape outputted by the model.

        Args:
            y (list or numpy.array): Input.

        Return:
            new_y (numpy.array): Reshaped output.
        """
        return np.array(list(y)).reshape(self.__y_shape)

    def __equalize(self, dataset, ratio_validation, ratio_testing):
        """Equalizes a dataset (list of tuples) such that it is uniformly
        distributed.

        Returns:
            training_samples (list of tuples): x and y for training.
            validation_samples (list of tuples): x and y for validation.
            testing_samples (list of tuples): x and y for testing.
        """
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
