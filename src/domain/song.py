
class Song:
    """Contains information about a song.

    Attributes:
        path (str): Path of the song.
        title (str): Title of the song.
        audio_format (str): Audio audio_format of the song (wav, mp3, etc.)
        duration (int): Duration of the song (in seconds).
        waveform (numpy.array): Waveform of the song.
    """

    def __init__(self, path, title, audio_format, duration, waveform):
        self.__path = path
        self.__title = title
        self.__format = audio_format
        self.__duration = duration
        self.__waveform = waveform

    @property
    def path(self):
        return self.__path

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
    def waveform(self):
        return self.__waveform
