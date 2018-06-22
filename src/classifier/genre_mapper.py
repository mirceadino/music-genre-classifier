class GenreMapper:
    """Converts the genre between its string form (genre), vector form (y),
    and index form (label). Genres are sorted.

    Example:
        Consider three genres: electro, latino, rock.
        "electro" has y [1, 0, 0] and label 0.
        "latino" has y [0, 1, 0] and label 1.
        "rock" has y [0, 0, 1] and label 2.

    Attributes:
        genres (list of str): Genres.
        labels (list of int): Labels/Indexes.
    """

    def __init__(self, genres):
        """Creates the mapper.

        Args:
            genres (list of str): List of genres.
        """
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
        """Constructs the mapper from the list of genres and labels.
        """
        for label in self.labels:
            self.__label_to_genre[label] = self.genres[label]
            self.__genre_to_label[self.genres[label]] = label

    def label_to_genre(self, label):
        if label not in self.labels:
            return None
        return self.__label_to_genre[label]

    def genre_to_label(self, genre):
        if genre not in self.genres:
            return None
        return self.__genre_to_label[genre]

    def label_to_y(self, label):
        return [0 if label != i else 1 for i in self.labels]

    def genre_to_y(self, genre):
        label = self.genre_to_label(genre)
        return [0 if label != i else 1 for i in self.labels]

    def y_to_genre(self, y):
        label = list(y).index(1)
        return self.label_to_genre(label)

    def y_to_label(self, y):
        label = list(y).index(1)
        return label
