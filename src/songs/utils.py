import csv

import librosa
import re
import scipy.io.wavfile
import subprocess

from src.songs.song import Song


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
    f, t, Sxx = scipy.signal.spectrogram(waveform, frames_per_sec,
                                         nperseg=frames_per_segment)
    Sxx = librosa.feature.melspectrogram(S=Sxx)
    return Sxx


def song_to_spectrogram_slices(song, rate, size, overlap=0):
    # TODO: Add documentation for this method.
    return song_to_slices(waveform_to_spectrogram(song, rate), size, overlap)


def read_songs_from_csv(path):
    # TODO: Add documentation about the method.
    songs = []
    with open(path, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            songs.append(Song.import_from_dictionary(row))
    return songs


def read_song_from_wav(path):
    # TODO: Add documentation about the method.
    frames_per_sec, waveform = scipy.io.wavfile.read(path)
    try:
        waveform = waveform[:, 0]
    except IndexError:
        waveform = waveform[:]
    return waveform, frames_per_sec


def song_id_from_yt_url(url):
    match = re.search(
        r"((?<=(v|V)/)|(?<=be/)|(?<=(\?|\&)v=)|(?<=embed/))([\w-]+)", url)
    if match:
        result = match.group(0)
    else:
        result = ""
    return result


def get_clean_yt_url(url):
    return "https://www.youtube.com/watch?v=" + song_id_from_yt_url(url)


def download_yt_song(url, output_directory="/tmp", audio_format="wav"):
    if output_directory[-1] != '/':
        output_directory += "/"

    url = get_clean_yt_url(url)

    output_filename_format = output_directory + '%(id)s.%(ext)s'
    subprocess.call("youtube-dl --extract-audio --audio-format {0} "
                    "--output {1} --quiet {2}"
                    .format(audio_format, output_filename_format, url).split())
    path = output_directory + song_id_from_yt_url(url) + "." + audio_format
    return path
