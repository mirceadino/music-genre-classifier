# General
NAME = "gtzan-10"
ROOT_MODEL = "resources/model/" + NAME + "/"
ROOT_DATASET = "resources/dataset/" + NAME + "/"
ROOT_SONGS = "resources/raw_dataset/" + NAME + "/"

# Classifier
GENRES = sorted(["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"])

# Model
NUM_CLASSES = len(GENRES)
SLICE_HEIGHT = 128
SLICE_WIDTH = 128
SLICE_OVERLAP = 64
PATH_MODEL = ROOT_MODEL + "model.tflearn"

# Training
BATCH_SIZE = 128
SHUFFLE = True
SNAPSHOT_EPOCH = True
SNAPSHOT_STEP = None
SHOW_METRIC = True

# Dataset
PATH_TRAINING_DATASET = ROOT_DATASET + "training"
PATH_VALIDATION_DATASET = ROOT_DATASET + "validation"
PATH_TESTING_DATASET = ROOT_DATASET + "testing"
RATIO_VALIDATION = 0.10
RATIO_TESTING = 0.10

# Songs
PATH_SONGS = ROOT_SONGS
PATH_SONG_INFO = ROOT_SONGS + "info.csv"

