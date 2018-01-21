import librosa
from IPython.display import Audio, display
import numpy as np
import matplotlib.pyplot as plt
# import style_transfer_2d_signal


# every spectrogram is truncated to a certain num of frames
N_FRAMES = 400

N_RECONSTRUCTION_ITER = 100


def read_audio(filename):
    signal, fs = librosa.load(filename)
    return signal, fs


def spectrum(signal):
    return np.log1p(np.abs(librosa.stft(signal)))


def trim(signal):
    return signal[:, :N_FRAMES]


def reconstruct(spectrogram, waveform_shape):
    reconstructed_waveform = np.random.random_sample(waveform_shape)
    for i in range(N_RECONSTRUCTION_ITER):
        reconstructed_waveform = librosa.istft(spectrogram * np.exp(1j *
                np.angle(librosa.stft(reconstructed_waveform)[:, :N_FRAMES])))
        if i%10 == 0:
            print('iteration {}: reconstructed_waveform is {}'.format(i, reconstructed_waveform))
    return reconstructed_waveform


def write_to_file(filename, signal, fs):
    """
    saves an audio signal to a wav file
    :param filename: path to output file
    :param signal:
    :param fs: sample rate
    :return:
    """
    librosa.output.write_wav(filename, signal, fs)


def show(spectrograms):
    """
    Plots a list of spectrograms
    :param spectrograms: a dict of spetrograms with their names, e.g.
                            {'content': content_spectrogram,
                               'style': style_spectrogram,
                                 'out': out_spectrogram}
    :return:
    """

    plt.figure(figsize=(10, 5))

    n_spectrograms = len(spectrograms)
    i = 1
    for name, spectrogram in spectrograms.items():
        plt.subplot(1, n_spectrograms, i)
        plt.title(name)
        plt.imshow(spectrogram)
        i = i + 1
    plt.show()

#
# content_waveform, fs = read_audio(CONTENT_FILENAME)
# style_waveform, _ = read_audio(STYLE_FILENAME)
#
# content_spectrogram = trim(spectrum(content_waveform))
# style_spectrogram = trim(spectrum(style_waveform))
#
# out_waveform = reconstruct(style_spectrogram, style_waveform.shape)
# write_to_file(OUTPUT_FILENAME, out_waveform, fs)
#
# spectrograms = {'content': content_spectrogram,
#                 'style': style_spectrogram,
#                 'out': style_spectrogram}
# show(spectrograms)
#
