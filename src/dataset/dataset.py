import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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
        logger.info("[+] Loading dataset \"{0}\"...".format(self.__name))
        if self.is_loaded:
            logger.warning("[+] Dataset \"{0}\" was already loaded.".format(self.__name))
            return False

        # TODO: Implement loading.
        self.__is_loaded = True
        logger.info("[+] Dataset \"{0}\" loaded!".format(self.__name))
        return self.__is_loaded

    def get(self):
        """Retrieves the whole dataset. Loads the dataset if not loaded.

        Returns:
            A tuple of numpy.array's of form (x, y).
        """
        if not self.is_loaded:
            self.load()

        # TODO: Implement retrieval.
        return None, None
