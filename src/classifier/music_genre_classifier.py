import numpy as np

from config import config
from src.utils.song_utils import song_to_spectrogram_slices


class MusicGenreClassifier:
    """Classifies a song into one of the existing genres based on a machine
    learning algorithm.

    The classifier wraps the algorithm (e.g. neural network), converts the raw
    input to suitable input for the algorithm and interprets the output.
    """

    def __init__(self, nn, genres):
        # TODO: Add documentation for this method.
        self.__model = nn
        self.__genres = sorted(genres)
        self.__genre_to_label = {}
        self.__label_to_genre = {}
        self.__build_mapping()

    def __build_mapping(self):
        # TODO: Add documentation for this method.
        genres = self.__genres
        for label in range(len(genres)):
            self.__label_to_genre[label] = genres[label]
            self.__genre_to_label[genres[label]] = label

    def label_to_genre(self, label):
        # TODO: Add documentation for this method.
        if label not in self.__label_to_genre.keys():
            return None
        return self.__label_to_genre[label]

    def genre_to_label(self, genre):
        # TODO: Add documentation for this method.
        if genre not in self.__genre_to_label.keys():
            return None
        return self.__genre_to_label[genre]

    @staticmethod
    def y_to_label(y):
        # TODO: Add documentation for this method.
        try:
            label = list(y).index(1)
            return label
        except ValueError:
            return None

    def predict(self, song, rate):
        # TODO: Add documentation for this method.
        slices = song_to_spectrogram_slices(song, rate, config.SLICE_SIZE,
                                            config.SLICE_OVERLAP)
        slices = np.array(slices).reshape([-1, 128, config.SLICE_SIZE, 1])

        count_per_label = {None: 0}
        for label in self.__label_to_genre.keys():
            count_per_label[label] = 0

        predictions = self.__model.predict_label(slices)
        for prediction in predictions:
            label = self.y_to_label(prediction)
            count_per_label[label] += 1

        count_per_genre = {}
        for label, count in count_per_label.items():
            count_per_genre[self.label_to_genre(label)] = count

        return count_per_genre
