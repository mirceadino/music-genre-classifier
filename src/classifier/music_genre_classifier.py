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

    def predict_song_from_counting(self, counting):
        return max(counting.items(), key=lambda pair: pair[1])[0]

    def predict_song_from_slices(self, slices):
        return self.predict_song_from_counting(
            self.predict_counting_from_slices(slices))

    def predict_song_from_raw_song(self, song, rate):
        return self.predict_song_from_counting(
            self.predict_counting_from_raw_song(song, rate))

    def count_predictions(self, predictions):
        count_per_label = {}
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

    def test(self, songs, y, batch=128):
        # Merge all song slices into one vector of slices.
        slices = []
        for song in songs:
            slices.extend(song)
        # Make batch predictions for all the slices.
        all_predictions = []
        for i in range(0, len(slices), batch):
            j = min(len(slices), i + batch)
            all_predictions.extend(self.__nn.predict_label(slices[i:j]))
        # Separate back the predictions per song.
        predictions_per_song = []
        i = 0
        for song in songs:
            j = i + len(song)
            predictions_per_song.append(all_predictions[i:j])
            i = j
        # Compute all genre predictions per song.
        genre_pred = []
        for predictions in predictions_per_song:
            counting = self.count_predictions(predictions)
            genre_pred.append(
                max(counting.items(), key=lambda pair: pair[1])[0])
        # Compute accuracy.
        num_matches = 0
        for i in range(len(songs)):
            genre = self.__genre_mapper.y_to_genre(y[i])
            prediction = genre_pred[i]
            if genre == prediction:
                num_matches += 1
        return num_matches / len(songs), genre_pred
