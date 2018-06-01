import numpy as np


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
        genres = list(map(lambda y: mapper.label_to_genre(np.argsort(y)[-1]), self.__y))
        genre_to_count = {}
        for genre in mapper.genres:
            genre_to_count[genre] = genres.count(genre)
        genre_to_percentage = {}
        for genre in mapper.genres:
            genre_to_percentage[genre] = genre_to_count[genre] / len(self.__y)
        print("[+] Frequency (count) of each genre: {0}".format(genre_to_count))
        print("[+] Frequency (percentage) of each genre: {0}".format(genre_to_percentage))

