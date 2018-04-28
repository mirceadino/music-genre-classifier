import logging
import pickle
import numpy as np


class Dataset:
    """Accessor for a dataset.

    Attributes:
        path (str): Path to the dataset.
        is_loaded (bool): Indicates whether the dataset was loaded or not.
    """

    def __init__(self, name, path):
        """Creates a dataset accessor.

        Args:
            name (str): Name of the dataset.
            path (str): Path to the stored dataset.
        """
        self.__name = name
        self.__path = path
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

    def load(self):
        """Loads the dataset. If dataset was previously loaded, it doesn't load
        it again.

        Returns:
            bool: True if the dataset was successfully loaded, False otherwise.
        """
        # TODO: Throw exception if loading fails instead of returning boolean.
        logging.info("[+] Loading dataset \"{0}\"...".format(self.__name))
        if self.is_loaded:
            logging.warning("[+] Dataset \"{0}\" was already loaded.".format(self.__name))
            return False

        with open(self.__path, "rb") as infile:
            dataset = pickle.load(infile)
            x, y = zip(*dataset)
            self.__x = np.array(list(x)).reshape([-1, 128, 128, 1])
            self.__y = np.array(list(y)).reshape([-1, 2])

        self.__is_loaded = True
        logging.info("[+] Dataset \"{0}\" loaded!".format(self.__name))
        return self.__is_loaded

    def get(self):
        """Retrieves the whole dataset. Loads the dataset if not loaded.

        Returns:
            A tuple of numpy.array's of form (x, y).
        """
        if not self.is_loaded:
            self.load()

        return self.__x, self.__y
