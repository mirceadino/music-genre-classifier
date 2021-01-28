# music-genre-classifier
:notes:

## About

This is the practical part of my project for my Bachelor Thesis ([abstract](https://drive.google.com/file/d/1wViCTlQ3BarVnTXCXDQOEXc3IgCep4yj/view?usp=sharing), [thesis](https://drive.google.com/file/d/1HAQACTiPUJvISyhBfj8pgcjJ6xieqJdI/view?usp=sharing)). We proposed an approach for music genre classification with deep learning based on audio signal processing. It consists of training a convolutional neural network on slices of the mel-spectrogram of songs and developing a classifier that would encapsulate the trained model and make final predictions.

We developed a library and a web application that illustrate our approach. The library wraps the model and the classifier and provides some utilities for data gathering and data processing. It also contains a command-line tool which supports some basic functions: converting the dataset into an adequate form for the classifier, training the model, evaluating it, and predicting individual songs. The web application is designed as a client-server model and can be used to perform online predictions using a trained classifier.

## Install and requirements

The application can be run on Linux or UNIX. Requirements that need to be installed beforehand:
- `Python 3.6` and `pip`
- `ffmpeg`
- `youtube-dl`

After installing `python3` and `pip`, install the packages from `requirements.txt`:
```
pip install -r requirements.txt
```
You might need to install some requirements separately, so watch for any error messages.

## How to use

A config file must be provided. See `config/config.py` for an example. You will also need a song dataset in WAV format and an CSV file that contains metadata about the songs: id, URL, title, duration, filename, audio_format, genre.

#### Command-line tool

Give permissions to `run_tool.sh` to run as executable.

```
chmod +x run_tool.sh
```

After that, you can use to command-line tool as follows:

```
./run_tool.sh --create [--no_equalize]
```
This uses the `DatasetCreator` to convert the raw songs into a dataset adequate to feed
the model. When the optional `--no_equalize` flag is set, the creator doesn’t enforce the
uniform distribution of the data in the dataset (by default, i.e. without setting this flag, the
creator might randomly discard some data in order to obtain an uniformly distributed dataset).

```
./run_tool.sh --train <n> [--resume]
```
This trains a `NeuralNetwork` for `n` epochs and saves the model at the end of the training on
the disk. When the optional `--resume` flag is set, the program loads the last saved model and
continues training on it. This option comes handy when doing fine tuning. In this mode, we
can view the progress of the training.

```
./run_tool.sh --test {testing, validation, training}
```

This computes the accuracy and the confusion matrix on the song predictions on the specified
dataset with the last saved model encapsulated in the `MusicGenreClassifier`.

```
./run_tool.sh --predict <path or url>
```
This uses the `MusicGenreClassifier` to predict the genre of a song stored on the specified
path. If the argument is detected to be a Youtube URL, it predicts the genre of that song.

#### Client-server web application

Give permissions to `run_server.sh` to run as executable.

```
chmod +x run_server.sh
```

After that, you can run the server:

```
./run_server.sh
```

Usually the server is run on port 5000. If this doesn't happen, either enforce to run it on port 5000 or modify the code in the client.

You can access the client by opening `client/index.html`.
