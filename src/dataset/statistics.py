from config import config
from src.classifier.music_genre_classifier import MusicGenreClassifier


class DatasetStatistics:
    def __init__(self, dataset):
        self.__dataset = dataset
        self.__x, self.__y = self.__dataset.get()

    def num_slices(self):
        print("[+] Number of slices: {0}".format(len(self.__x)))

    def slices_per_genre(self):
        classifier = MusicGenreClassifier(None, config.GENRES)
        genres = list(map(lambda y: classifier.label_to_genre(MusicGenreClassifier.y_to_label(y)), self.__y))
        genre_to_count = {}
        for genre in config.GENRES:
            genre_to_count[genre] = genres.count(genre)
        genre_to_percentage = {}
        for genre in config.GENRES:
            genre_to_percentage[genre] = genre_to_count[genre] / len(self.__y)
        print("[+] Frequency (count) of each genre: {0}".format(genre_to_count))
        print("[+] Frequency (percentage) of each genre: {0}".format(genre_to_percentage))

