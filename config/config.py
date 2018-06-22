# Classifier
GENRES = sorted(["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"])
GENRES = sorted(["latino", "electro", "classical"])

# Model
NUM_CLASSES = len(GENRES)
SLICE_HEIGHT = 128
SLICE_WIDTH = 128
SLICE_OVERLAP = 64
PATH_MODEL = "resources/model/gtzan-10-model-yy.tflearn"

# Training
BATCH_SIZE = 128
SHUFFLE = True 
SNAPSHOT_EPOCH = True
SNAPSHOT_STEP = None
SHOW_METRIC = True

# Dataset
PATH_TRAINING_DATASET = "resources/dataset/gtzan-10-training-yy"
PATH_VALIDATION_DATASET = "resources/dataset/gtzan-10-validation-yy"
PATH_TESTING_DATASET = "resources/dataset/gtzan-10-testing-yy"
RATIO_VALIDATION = 0.27
RATIO_TESTING = 0.10

# Songs
PATH_SONGS = "resources/raw_dataset/"
PATH_SONG_INFO = "resources/raw_dataset/info.csv"
