# Classifier
GENRES = sorted(["classical", "rock", "latino", "electro"])

# Model
NUM_CLASSES = len(GENRES)
SLICE_HEIGHT = 128
SLICE_WIDTH = 128
SLICE_OVERLAP = 64
PATH_MODEL = "resources/model/yt/model-01.tflearn"

# Training
BATCH_SIZE = 128
SHUFFLE = True 
SNAPSHOT_EPOCH = True
SNAPSHOT_STEP = None
SHOW_METRIC = True

# Dataset
PATH_TRAINING_DATASET = "resources/dataset/yt/training-01"
PATH_VALIDATION_DATASET = "resources/dataset/yt/validation-01"
PATH_TESTING_DATASET = "resources/dataset/yt/testing-01"
RATIO_VALIDATION = 0.10
RATIO_TESTING = 0.10

# Songs
PATH_SONGS = "resources/raw_dataset/yt/"
PATH_SONG_INFO = "resources/raw_dataset/yt/info.csv"
