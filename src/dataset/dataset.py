import logging
import pickle


class Dataset:
    """Accessor for a dataset.

    Attributes:
        name (str): Name of the dataset.
        path (str): Path to the dataset.
        is_loaded (bool): Indicates whether the dataset was loaded or not.
    """

    def __init__(self, name, path, dataset_creator):
        """Creates a dataset accessor.

        Args:
            name (str): Name of the dataset.
            path (str): Path to the stored dataset.
            x_shape (list): Shape of x. Example: [-1, 128, 128, 1].
            y_shape (list): Shape of y. Example: [-1, 4].
        """
        self.__name = name
        self.__path = path
        self.__dataset_creator = dataset_creator
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

        Returns:
            bool: True if the dataset was successfully loaded, False otherwise.
        """
        # TODO: Throw exception if loading fails instead of returning bool.
        logging.info("[+] Loading dataset \"{0}\"...".format(self.__name))

        self.__is_loaded = False

        with open(self.__path, "rb") as infile:
            dataset = pickle.load(infile)
            if extend:
                extended_dataset = []
                for slices, label in dataset:
                    for slice in slices:
                        extended_dataset.append((slice, label))
                x, y = zip(*extended_dataset)
                self.__x = self.__dataset_creator.reshape_x(x)
                self.__y = self.__dataset_creator.reshape_y(y)
            else:
                x, y = zip(*dataset)
                self.__x = []
                for song in x:
                    self.__x.append(self.__dataset_creator.reshape_x(song))
                self.__y = self.__dataset_creator.reshape_y(y)

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
