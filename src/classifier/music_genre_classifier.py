class MusicGenreClassifier:
    """Classifies a song into one of the existing genres based on a machine
    learning algorithm.

    The classifier wraps the algorithm (e.g. neural network), converts the raw
    input to suitable input for the algorithm and interprets the output.
    """

    def __init__(self, nn, genre_mapper, dataset_creator):
        # TODO: Add documentation for this method.
        self.__nn = nn
        self.__genre_mapper = genre_mapper
        self.__dataset_creator = dataset_creator

    def predict(self, song, rate):
        # TODO: Add documentation for this method.
        slices = self.__dataset_creator.song_to_x(song, rate)

        count_per_label = {None: 0}
        for label in self.__genre_mapper.labels:
            count_per_label[label] = 0

        predictions = self.__nn.predict_label(slices)
        for prediction in predictions:
            label = prediction[0]
            count_per_label[label] += 1

        count_per_genre = {}
        for label, count in count_per_label.items():
            count_per_genre[self.__genre_mapper.label_to_genre(label)] = count

        return count_per_genre
