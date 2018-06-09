# Classifier
GENRES = sorted(["electro", "classical", "latino"])

# Model
NUM_CLASSES = len(GENRES)
SLICE_SIZE = 128
SLICE_OVERLAP = 64
PATH_MODEL = "model/model-04.tflearn"

# Training
BATCH_SIZE = 128
SHUFFLE = False
SNAPSHOT_EPOCH = True
SNAPSHOT_STEP = None
SHOW_METRIC = True

# Dataset
PATH_TRAINING_DATASET = "dataset/training-03"
PATH_VALIDATION_DATASET = "dataset/validation-03"
PATH_TESTING_DATASET = "dataset/testing-03"
RATIO_VALIDATION = 0.27
RATIO_TESTING = 0.1

# Songs
PATH_SONGS = "raw_dataset/"
PATH_SONG_INFO = "raw_dataset/info_tiny.csv"
