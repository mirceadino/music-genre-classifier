class Song:
    """Contains information about a song.

    Attributes:
        id (str): ID of the song.
        title (str): Title of the song.
        audio_format (str): Audio audio_format of the song (wav, mp3, etc.)
        duration (int): Duration of the song (in seconds).
        genre (str): Label for the song.
    """

    def __init__(self, song_id, title, audio_format, duration, genre):
        self.__id = song_id
        self.__title = title
        self.__format = audio_format
        self.__duration = duration
        self.__genre = genre

    @staticmethod
    def import_from_dictionary(dictionary):
        """Imports a song from a dictionary.

        Args:
            dictionary (dict): Dictionary of pairs of (attribute, value).

        Returns:
            Song: created song from the dictionary.
        """
        song = Song(None, None, None, None, None)
        song.__id = dictionary["id"]
        song.__title = dictionary["title"]
        song.__format = dictionary["audio_format"]
        song.__duration = dictionary["duration"]
        song.__genre = dictionary["genre"]
        return song

    @property
    def id(self):
        return self.__id

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
    def genre(self):
        return self.__genre
