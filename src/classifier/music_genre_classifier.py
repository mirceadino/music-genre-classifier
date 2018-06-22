class MusicGenreClassifier:
    """Classifies a song into one of the existing genres by using a machine
    learning algorithm.

    The classifier wraps the algorithm (e.g. neural network), converts the raw
    input to suitable input for the algorithm and performs the final prediction.
    """

    def __init__(self, nn, mapper, creator):
        """Creates a classifier.

        Args:
            nn (nn.neural_network.NeuralNetwork): Model to be used.
            mapper (classifier.genre_mapper.GenreMapper): Mapper for genres.
            creator (dataset.dataset_creator.DatasetCreator): Creator of
            datasets and inputs for the model.
        """
        self.__nn = nn
        self.__mapper = mapper
        self.__creator = creator

    def predict_song_from_counting(self, counting):
        """Predicts the genre of a song from the votes for each genre from
        the slice predictions.

        Args:
            counting (dict): Dictionary of votes. Key must be the name of the
            genre (e.g. "latino"), value must be the number of votes.

        Returns:
            genre (str): Final prediction.
        """
        return max(counting.items(), key=lambda pair: pair[1])[0]

    def predict_song_from_slices(self, slices):
        """Predicts the genre of a song from slices.

        Args:
            slices (list of numpy.array): Slices of the song.

        Returns:
            genre (str): Final prediction.
        """
        return self.predict_song_from_counting(
            self.predict_counting_from_slices(slices))

    def predict_song_from_raw_song(self, waveform, rate):
        """Predicts the genre of a song from the waveform by converting it
        into suitable input for the model and making a final prediction.

        Args:
            waveform (list or numpy.array): Waveform of the song.
            rate (int): Sampling rate of the waveform.

        Returns:
            genre (str): Final prediction.
        """
        return self.predict_song_from_counting(
            self.predict_counting_from_raw_song(waveform, rate))

    def count_predictions(self, predictions):
        """Counts the votes for each genre.

        Args:
            predictions (list of numpy.array): Predictions.

        Returns:
            counting (dict): Dictionary of votes. Key is a genre as string (
            e.g. "latino"), value is the number of votes.
        """
        count_per_label = {}
        for label in self.__mapper.labels:
            count_per_label[label] = 0

        for prediction in predictions:
            label = prediction[0]
            count_per_label[label] += 1

        count_per_genre = {}
        for label, count in count_per_label.items():
            count_per_genre[self.__mapper.label_to_genre(label)] = count

        return count_per_genre

    def predict_counting_from_slices(self, slices):
        """Makes predictions for all the slices and returns the number of
        votes for each genre.

        Args:
            slices (list of numpy.array): Slices of the song.

        Returns:
            counting (dict): Dictionary of votes. Key is a genre as string (
            e.g. "latino"), value is the number of votes.
        """
        predictions = self.__nn.predict_label(slices)
        return self.count_predictions(predictions)

    def predict_counting_from_raw_song(self, waveform, rate):
        """Makes predictions for all the slices and returns the number of
        votes for each genre.

        Args:
            waveform (list or numpy.array): Waveform of the song.
            rate (int): Sampling rate of the waveform.

        Returns:
            counting (dict): Dictionary of votes. Key is a genre as string (
            e.g. "latino"), value is the number of votes.
        """
        slices = self.__creator.song_to_x(waveform, rate)
        predictions = self.__nn.predict_label(slices)
        return self.count_predictions(predictions)

    def test(self, songs, y, batch=128):
        """Evaluates the model on a set of samples and based on the correct
        labels.

        Args:
            songs (list of numpy.array): List of songs represented as a list
            of slices as created by the DatasetCreator.
            y (list of numpy.array): Correct labels of the songs in vector
            form.
            batch (int): Size of the batch. Default: 128

        Return:
            accuracy (float): Number of correct predictions divided by the
            total number of songs.
            predictions (str): Predictions for each song as string.
        """
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
            genre = self.__mapper.y_to_genre(y[i])
            prediction = genre_pred[i]
            if genre == prediction:
                num_matches += 1
        return num_matches / len(songs), genre_pred
