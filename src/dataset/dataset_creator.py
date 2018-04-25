
class DatasetCreator:
    """Converts raw data to data that can be used for training, validating and
    testing the learning model.
    """

    def __init__(self):
        """Creates a dataset creator."""
        pass

    def create(self, path_raw, path_training, path_validation, path_testing,
               ratio_validation=0, ratio_testing=0):
        """Converts raw data to data that can be fed to the classifier.

        Args:
            path_raw (str): Path where the raw data is stored.
            path_training (str): Path where the training dataset will be stored.
            path_validation (str): Path where the cross-validation dataset will
                be stored.
            path_testing (str): Path where the testing dataset will be stored.
            ratio_validation (double): Ratio of the size of the cross-validation
                dataset. Must be between 0 and 1. Defaults is 0.
            ratio_testing (double): Ratio of the size of the testing dataset.
                Must be between 0 and 1. Defaults is 0.

        Returns:
            bool: True if successful. False otherwise.
        """
        # TODO: Implement dataset creation.
        return True
