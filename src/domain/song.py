import numpy

from config import config


class Song:
    """Contains information about a song.

    Attributes:
        id (str): ID of the song.
        filename (str): Path of the song.
        title (str): Title of the song.
        audio_format (str): Audio audio_format of the song (wav, mp3, etc.)
        duration (int): Duration of the song (in seconds).
        labels (list): List of labels for the song.
        waveform (numpy.array): Waveform of the song.
    """

    def __init__(self, song_id, filename, title, audio_format, duration, labels):
        # TODO: Remove unnecessary attributes: filename, waveform.
        self.__id = song_id
        self.__filename = filename
        self.__title = title
        self.__format = audio_format
        self.__duration = duration
        self.__labels = labels
        self.__validate()

    @staticmethod
    def import_from_dictionary(dictionary):
        song = Song(None, None, None, None, None, None)
        song.__id = dictionary["id"]
        song.__filename = dictionary["filename"]
        song.__title = dictionary["title"]
        song.__format = dictionary["audio_format"]
        song.__duration = dictionary["duration"]
        song.__labels = dictionary["labels"].split()
        song.__validate()
        return song

    @property
    def id(self):
        return self.__id

    @property
    def filename(self):
        return self.__filename

    @property
    def title(self):
        return self.__title

    @property
    def audio_format(self):
        return self.__format

    @property
    def duration(self):
        return self.__duration

    @property
    def labels(self):
        return self.__labels

    def get_y(self):
        y = [1 if genre in self.labels else 0 for genre in config.GENRES]
        return numpy.array(y)

    def __validate(self):
        if self.labels is not None:
            for label in self.labels:
                if label not in config.GENRES:
                    raise ValueError("Label '{0}' is not present in {1}.".format(label, config.GENRES))
