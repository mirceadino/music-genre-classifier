
class Dataset:
    """Accessor for a dataset.

    Attributes:
        path (str): Path to the dataset.
        is_loaded (bool): Indicates whether the dataset was loaded or not.
    """

    def __init__(self, path):
        """Creates a dataset accessor.

        Args:
            path (str): Path to the stored dataset.
        """
        self.__path = path
        self.__is_loaded = False

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
        if self.is_loaded:
            return False

        # TODO: Implement loading.
        self.__is_loaded = True
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
