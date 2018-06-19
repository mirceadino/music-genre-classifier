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

    def predict_from_slices(self, slices):
        counting = self.predict_counting_from_slices(slices)
        return max(counting.items(), key=lambda pair: pair[1])[0]

    def predict_from_raw_song(self, song, rate):
        counting = self.predict_counting_from_raw_song(song, rate)
        return max(counting.items(), key=lambda pair: pair[1])[0]

    def count_predictions(self, predictions):
        count_per_label = {None: 0}
        for label in self.__genre_mapper.labels:
            count_per_label[label] = 0

        for prediction in predictions:
            label = prediction[0]
            count_per_label[label] += 1

        count_per_genre = {}
        for label, count in count_per_label.items():
            count_per_genre[self.__genre_mapper.label_to_genre(label)] = count

        return count_per_genre

    def predict_counting_from_slices(self, slices):
        # TODO: Add documentation for this method.
        predictions = self.__nn.predict_label(slices)
        return self.count_predictions(predictions)

    def predict_counting_from_raw_song(self, song, rate):
        # TODO: Add documentation for this method.
        slices = self.__dataset_creator.song_to_x(song, rate)
        predictions = self.__nn.predict_label(slices)
        return self.count_predictions(predictions)

    def test(self, slices, labels):
        all_predictions = self.__nn.predict_label(slices)
        slices_per_song = len(slices) // len(labels)
        matches = 0
        total = len(labels)
        all_genres_pred = []
        for song_index in range(len(labels)):
            i = song_index * slices_per_song
            j = (song_index + 1) * slices_per_song
            predictions = all_predictions[i:j]
            predictions = self.count_predictions(predictions)
            prediction = max(predictions.items(), key=lambda pair: pair[1])[0]
            genre = self.__genre_mapper.y_to_genre(labels[song_index])
            if genre == prediction:
                matches += 1
            all_genres_pred.append(prediction)
        return matches / total, all_genres_pred
