import csv

import librosa
import scipy.io.wavfile

from src.domain.song import Song


def song_to_slices(song, size, overlap=0):
    # TODO: Add documentation for this method.
    slices = []
    begin = 0
    while begin + size < len(song[0]):
        slices.append(song[:, begin:begin + size])
        begin += size - overlap
    return slices


def waveform_to_spectrogram(waveform, frames_per_sec):
    # TODO: Add documentation for this method.
    frames_per_segment = frames_per_sec // 43  # TODO: Correctly choose the number here.
    f, t, Sxx = scipy.signal.spectrogram(waveform, frames_per_sec, nperseg=frames_per_segment)
    Sxx = librosa.feature.melspectrogram(S=Sxx)
    return Sxx


def song_to_spectrogram_slices(song, rate, size, overlap=0):
    # TODO: Add documentation for this method.
    return song_to_slices(waveform_to_spectrogram(song, rate), size, overlap)


def read_songs_from_csv(path):
    songs = []
    with open(path, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            songs.append(Song.import_from_dictionary(row))
    return songs


def read_song_from_wav(path):
    frames_per_sec, waveform = scipy.io.wavfile.read(path)
    try:
        waveform = waveform[:, 0]
    except IndexError:
        waveform = waveform[:]
    return waveform, frames_per_sec
