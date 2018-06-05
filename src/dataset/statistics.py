import numpy as np
import pandas as pd


class DatasetStatistics:
    # TODO: Add documentation about the class.

    def __init__(self, dataset, genre_mapper):
        # TODO: Add documentation about the method.
        self.__dataset = dataset
        self.__genre_mapper = genre_mapper
        self.__x, self.__y = self.__dataset.get()

    def all(self):
        self.num_slices()
        self.slices_per_genre()

    def num_slices(self):
        # TODO: Add documentation about the method.
        print("[+] Number of slices: {0}".format(len(self.__x)))

    def slices_per_genre(self):
        # TODO: Add documentation about the method.
        mapper = self.__genre_mapper
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

        print("[+] Frequency of each genre:")
        print(pd.DataFrame(matrix).to_string())
        """
        print("[+] Frequency (count) of each genre:")
        print(pd.Series(genre_to_count).to_string())
        print("[+] Frequency (percentage) of each genre:")
        print(pd.Series(genre_to_percentage).to_string())
        """

    def confusion_matrix(self, y_pred):
        matrix = {}
        for expected_genre in self.__genre_mapper.genres:
            matrix[expected_genre] = {}
            for predicted_genre in self.__genre_mapper.genres:
                matrix[expected_genre][predicted_genre] = 0

        for i in range(len(self.__y)):
            expected_y = self.__y[i]
            predicted_y = y_pred[i]
            expected_genre = self.__genre_mapper.y_to_genre(expected_y)
            predicted_genre = self.__genre_mapper.y_to_genre(predicted_y)
            matrix[expected_genre][predicted_genre] += 1

        print("[+] Confusion matrix:")
        print(pd.DataFrame(matrix).to_string(na_rep='-'))
