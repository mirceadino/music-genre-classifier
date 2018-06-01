class GenreMapper:
    # TODO: Add documentation for this class.

    def __init__(self, genres):
        # TODO: Add documentation for this method.
        self.__genres = sorted(genres)
        self.__labels = range(len(genres))
        self.__genre_to_label = {}
        self.__label_to_genre = {}
        self.__build_mapping()

    @property
    def genres(self):
        return self.__genres

    @property
    def labels(self):
        return self.__labels

    def __build_mapping(self):
        # TODO: Add documentation for this method.
        for label in self.labels:
            self.__label_to_genre[label] = self.genres[label]
            self.__genre_to_label[self.genres[label]] = label

    def label_to_genre(self, label):
        # TODO: Add documentation for this method.
        if label not in self.labels:
            return None
        return self.__label_to_genre[label]

    def genre_to_label(self, genre):
        # TODO: Add documentation for this method.
        if genre not in self.genres:
            return None
        return self.__genre_to_label[genre]

    def genre_to_y(self, genre):
        label = self.genre_to_label(genre)
        return [0 if label != i else 1 for i in self.labels]
