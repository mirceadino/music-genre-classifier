# Model
NUM_CLASSES = 2
IMAGE_SIZE = 128
PATH_MODEL = "model/model.tflearn"

# Training
NUM_EPOCHS = 10
BATCH_SIZE = 128
SHUFFLE = True
SNAPSHOT_EPOCH = True
SNAPSHOT_STEP = 100
SHOW_METRIC = True

# Classifier
GENRES = sorted(["electro", "classical"])

# Dataset
PATH_TRAINING_DATASET = "dataset/training"
PATH_VALIDATION_DATASET = "dataset/validation"
PATH_TESTING_DATASET = "dataset/testing"
RATIO_VALIDATION = 0.3
RATIO_TESTING = 0.1

# Songs
PATH_SONGS = "raw_dataset/"
PATH_SONG_INFO = "raw_dataset/info.csv"
