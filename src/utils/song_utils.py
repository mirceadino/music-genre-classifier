import librosa


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
    Sxx = librosa.feature.melspectrogram(y=waveform, sr=frames_per_sec, n_fft=frames_per_segment)
    return Sxx


def song_to_spectrogram_slices(song, size, overlap=0):
    # TODO: Add documentation for this method.
    return song_to_slices(waveform_to_spectrogram(song), size, overlap)
