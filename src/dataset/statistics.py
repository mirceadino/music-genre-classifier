import numpy as np
import pandas as pd


class DatasetStatistics:
    """Prints statistics about a dataset and predictions. All stats are
    printed to stdout.
    """

    def __init__(self, dataset, mapper):
        """Creates a DatasetStatistics.

        Args:
            dataset (dataset.dataset.Dataset): Dataset for which we want stats.
            mapper (classifier.genre_mapper.GenreMapper): Mapper for genres.
        """
        self.__dataset = dataset
        self.__mapper = mapper
        self.__x, self.__y = self.__dataset.get()

    def all(self):
        """Print all stats.
        """
        self.num_slices()
        self.slices_per_genre()

    def num_slices(self):
        """Prints the number of samples.
        """
        print("Total number of items: {0}".format(len(self.__y)))

    def slices_per_genre(self):
        """Prints the distribution of the samples.
        """
        mapper = self.__mapper
        genres = list(
            map(lambda y: mapper.label_to_genre(np.argsort(y)[-1]), self.__y))
        genre_to_count = {}
        for genre in mapper.genres:
            genre_to_count[genre] = genres.count(genre)
        genre_to_percentage = {}
        for genre in mapper.genres:
            genre_to_percentage[genre] = genre_to_count[genre] / len(self.__y)

        matrix = {}
        for genre in mapper.genres:
            matrix[genre] = {'count': genre_to_count[genre], 'percent': \
                str(round(genre_to_percentage[genre], 4))}

        print("Frequency of each genre:")
        print(pd.DataFrame(matrix).to_string())

    def confusion_matrix(self, genres_pred):
        """Prints the confusion matrix given the predictions. It counts the
        matches and mismatches between the correct genres from the dataset
        and the predicted genres that are provided.

        Args:
            genres_pred (list of str): Predictions for each sample.
        """
        matrix = {}
        for expected_genre in self.__mapper.genres:
            matrix[expected_genre] = {}
            for predicted_genre in self.__mapper.genres:
                matrix[expected_genre][predicted_genre] = 0

        for i in range(len(self.__y)):
            expected_y = self.__y[i]
            expected_genre = self.__mapper.y_to_genre(expected_y)
            predicted_genre = genres_pred[i]
            matrix[expected_genre][predicted_genre] += 1

        print("Confusion matrix:")
        print(pd.DataFrame(matrix).to_string(na_rep='-'))
