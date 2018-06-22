# Classifier
GENRES = sorted(["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"])

# Model
NUM_CLASSES = len(GENRES)
SLICE_HEIGHT = 128
SLICE_WIDTH = 128
SLICE_OVERLAP = 64
PATH_MODEL = "resources/model/gtzan-10/model.tflearn"

# Training
BATCH_SIZE = 128
SHUFFLE = True 
SNAPSHOT_EPOCH = True
SNAPSHOT_STEP = None
SHOW_METRIC = True

# Dataset
PATH_TRAINING_DATASET = "resources/dataset/gtzan-10/training"
PATH_VALIDATION_DATASET = "resources/dataset/gtzan-10/validation"
PATH_TESTING_DATASET = "resources/dataset/gtzan-10/testing"
RATIO_VALIDATION = 0.10
RATIO_TESTING = 0.10

# Songs
PATH_SONGS = "resources/raw_dataset/gtzan-10/"
PATH_SONG_INFO = "resources/raw_dataset/gtzan-10/info.csv"
