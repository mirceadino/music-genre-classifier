import os
import logging

from flask import Flask, jsonify
from flask import abort
from flask import request

import tensorflow as tf

from config import config
from src.classifier.genre_mapper import GenreMapper
from src.classifier.music_genre_classifier import MusicGenreClassifier
from src.dataset.dataset_creator import DatasetCreator
from src.nn.neural_network import NeuralNetwork
from src.songs.utils import read_song_from_wav, download_yt_song

logging.getLogger().setLevel(logging.INFO)
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)


def get_classifier():
    genre_mapper = GenreMapper(config.GENRES)
    dataset_creator = DatasetCreator(genre_mapper,
                                     slice_height=config.SLICE_HEIGHT,
                                     slice_width=config.SLICE_WIDTH,
                                     slice_overlap=config.SLICE_OVERLAP)
    nn = NeuralNetwork(num_rows=config.SLICE_HEIGHT,
                       num_cols=config.SLICE_WIDTH,
                       num_classes=len(config.GENRES))
    classifier = MusicGenreClassifier(nn, genre_mapper, dataset_creator)
    return classifier


classifier = get_classifier()


@app.route('/predict', methods=['POST'])
def predict():
    if not request.json or not 'url' in request.json:
        abort(400)

    url = request.json['url']

    try:
        path = download_yt_song(url)
        song, rate = read_song_from_wav(path)
        genre = classifier.predict_song_from_raw_song(song, rate)
        counting = classifier.predict_counting_from_raw_song(song, rate)
        os.remove(path)

        return jsonify({"predicted_genre": genre, "counting": counting})

    except FileNotFoundError:
        abort(404)
