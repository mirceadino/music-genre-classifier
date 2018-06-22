import csv

import librosa
import re
import scipy.io.wavfile
import subprocess

from src.songs.song import Song


def song_to_slices(song, size, overlap=0):
    """Slices a song in whatever form it is into overlapping slices.

    Args:
        song (list): Song represented as a list (list of samples, list of
        lists etc.)
        size (int): Size of the slice.
        overlap (int): Overlap between the slices.

    Returns:
        slices (list of lists): Slices.
    """
    slices = []
    begin = 0
    while begin + size < len(song[0]):
        slices.append(song[:, begin:begin + size])
        begin += size - overlap
    return slices


def waveform_to_spectrogram(waveform, frames_per_sec):
    """Converts waveform to mel-spectrogram. By default, the height of the
    spectrogram will be 128.

    Args:
        waveform (list of int): Samples.
        frames_per_sec (int): Sampling rate.

    Returns:
        spectrogram (numpy.array): Mel-specteogram.
    """
    frames_per_segment = frames_per_sec // 43  # TODO: Correctly choose the number here.
    f, t, Sxx = scipy.signal.spectrogram(waveform, frames_per_sec,
                                         nperseg=frames_per_segment)
    Sxx = librosa.feature.melspectrogram(S=Sxx)
    return Sxx


def song_to_spectrogram_slices(waveform, rate, size, overlap=0):
    """Converts waveform to mel-spectrogram and then into slices.

    Args:
        waveform (list of int): Samples.
        rate (int): Sampling rate.
        size (int): Size of the slice.
        overlap (int): Overlap between the slices.

    Returns:
        slices (list of lists): Slices.
    """
    return song_to_slices(waveform_to_spectrogram(waveform, rate), size,
                          overlap)


def read_songs_from_csv(path):
    """Reads meta-information about songs from a .csv file.
    The file must have a header with the attributes: id, title, audio_format,
    duration, genre.
    The rows after the header must follow these attributes.

    Args:
        path (str): Path to the .csv file.

    Returns:
        songs (list of songs.song.Song): List of songs.
    """
    songs = []
    with open(path, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            songs.append(Song.import_from_dictionary(row))
    return songs


def read_song_from_wav(path):
    """Reads a .wav file.

    Args:
        path (str): Path to the .wav file.

    Returns:
        waveform (list): Samples.
        frames_per_sec (int): Sampling rate.
    """
    frames_per_sec, waveform = scipy.io.wavfile.read(path)
    try:
        waveform = waveform[:, 0]
    except IndexError:
        waveform = waveform[:]
    return waveform, frames_per_sec


def song_id_from_yt_url(url):
    """Extracts the song id from a Youtube URL.

    Args:
        url (str): Youtube URL.

    Returns:
        id (str): Extracted id.
    """
    match = re.search(
        r"((?<=(v|V)/)|(?<=be/)|(?<=(\?|\&)v=)|(?<=embed/))([\w-]+)", url)
    if match:
        result = match.group(0)
    else:
        result = ""
    return result


def get_clean_yt_url(url):
    """Extracts the song id from the Youtube URL and outpus an URL that
    contains only the domain and the URL.

    Example:
        For input "https://www.youtu.be/watch?v=9Pey-HmXGfs&t=0s&list=WL&index=4&t=39",
        it outputs "https://www.youtube.com/watch?v=9Pey-HmXGfs".

    Args:
        url (str): Youtube URL.

    Returns:
        clean_url (url): Clean Youtube URL.
    """
    return "https://www.youtube.com/watch?v=" + song_id_from_yt_url(url)


def download_yt_song(url, output_directory="/tmp", audio_format="wav"):
    """Downloads a Youtube song. The name of the resulting file will be
    <song_id>.<audio_format>.

    Args:
        url (str): Youtube URL.
        output_directory (str): Output directory. Default: "/tmp"
        audio_format (str): Desired audio format. Default: "wav"
    """
    if output_directory[-1] != '/':
        output_directory += "/"

    url = get_clean_yt_url(url)

    output_filename_format = output_directory + '%(id)s.%(ext)s'
    subprocess.call("youtube-dl --extract-audio --audio-format {0} "
                    "--output {1} --quiet {2}"
                    .format(audio_format, output_filename_format, url).split())
    path = output_directory + song_id_from_yt_url(url) + "." + audio_format
    return path
