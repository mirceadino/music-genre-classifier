import logging

from src.songs.utils import load_from_file


class Dataset:
    """Accessor for a dataset.

    We distinguish between two types of datasets: extended dataset and
    non-extended dataset.
    For x, in both cases, it is a list of samples.
    A non-extended dataset has as sample a list of numpy.array's. A sample
    corresponds to a song and a numpy.array to a slice of the song.
    An extended dataset has as sample a numpy.array which corresponds to a
    slice of a song.
    For y, in both cases, it is a list of numpy.array's, and each
    numpy.array corresponds to the genre in vector form of a sample.

    Datasets are stored in file in non-extended form, however they can be
    extended when loading into memory.

    Example:
         For a non-extended dataset:
         x = [[slice, slice, slice], [slice, slice, slice]] and y = [vector,
         vector].
         For an extended dataset:
         x = [slice, slice, slice, slice, slice, slice] and y = [vector,
         vector, vector, vector, vector, vector].

    Attributes:
        name (str): Name of the dataset.
        path (str): Path to the dataset.
        is_loaded (bool): Indicates whether the dataset was loaded or not.
    """

    def __init__(self, name, path, creator):
        """Creates a dataset accessor.

        Args:
            name (str): Name of the dataset.
            path (str): Path to the stored dataset.
            creator (dataset.dataset_creator.DatasetCreator): Creator of
            datasets and inputs for the model.
        """
        self.__name = name
        self.__path = path
        self.__creator = creator
        self.__is_loaded = False
        self.__x = None
        self.__y = None

    @property
    def name(self):
        return self.__name

    @property
    def path(self):
        return self.__path

    @property
    def is_loaded(self):
        return self.__is_loaded

    def load(self, extend=True):
        """Loads the dataset. If dataset was already loaded, it reloads it.

        Args:
            extend (bool): Whether to extend the dataset from songs to
            slices only. Default: True.
        """
        logging.info("[+] Loading dataset \"{0}\"...".format(self.__name))

        self.__is_loaded = False

        dataset = load_from_file(self.__path)
        if extend:
            extended_dataset = []
            for slices, label in dataset:
                for slice in slices:
                    extended_dataset.append((slice, label))
            x, y = zip(*extended_dataset)
            self.__x = self.__creator.reshape_x(x)
            self.__y = self.__creator.reshape_y(y)
        else:
            x, y = zip(*dataset)
            self.__x = []
            for song in x:
                self.__x.append(self.__creator.reshape_x(song))
            self.__y = self.__creator.reshape_y(y)

        self.__is_loaded = True
        logging.info("[+] Dataset \"{0}\" loaded!".format(self.__name))

    def get(self):
        """Retrieves the whole dataset. Loads the dataset if not loaded.

        Returns:
            x (list): List of samples.
            y (list): List of vectorial labels.
        """
        if not self.is_loaded:
            self.load()

        return self.__x, self.__y
