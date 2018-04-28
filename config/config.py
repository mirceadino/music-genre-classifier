# Classifier
GENRES = sorted(["latino", "classical"])

# Model
NUM_CLASSES = len(GENRES)
SLICE_SIZE = 128
SLICE_OVERLAP = 64
PATH_MODEL = "model/model.tflearn"

# Training
NUM_EPOCHS = 10
BATCH_SIZE = 128
SHUFFLE = False
SNAPSHOT_EPOCH = True
SNAPSHOT_STEP = 100
SHOW_METRIC = True

# Dataset
PATH_TRAINING_DATASET = "dataset/training"
PATH_VALIDATION_DATASET = "dataset/validation"
PATH_TESTING_DATASET = "dataset/testing"
RATIO_VALIDATION = 0.3
RATIO_TESTING = 0.1

# Songs
PATH_SONGS = "raw_dataset/"
PATH_SONG_INFO = "raw_dataset/info.csv"
