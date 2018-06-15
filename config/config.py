# Classifier
#GENRES = sorted(["electro", "classical", "latino"])
GENRES = sorted(["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"])
#GENRES = sorted(["classical", "disco", "hiphop", "jazz", "rock"])

# Model
NUM_CLASSES = len(GENRES)
SLICE_HEIGHT = 128
SLICE_WIDTH = 128
SLICE_OVERLAP = -64
PATH_MODEL = "model/model-cnn-02.tflearn"

# Training
BATCH_SIZE = 128
SHUFFLE = True 
SNAPSHOT_EPOCH = True
SNAPSHOT_STEP = None
SHOW_METRIC = True

# Dataset
PATH_TRAINING_DATASET = "dataset/training-genres"
PATH_VALIDATION_DATASET = "dataset/validation-genres"
PATH_TESTING_DATASET = "dataset/testing-genres"
RATIO_VALIDATION = 0.1
RATIO_TESTING = 0.1

# Songs
PATH_SONGS = "raw_dataset-genres/"
PATH_SONG_INFO = "raw_dataset-genres/info.csv"
