from style_transfer import *
from audio_utils import *
import sys


OUTPUT_FILENAME = 'outputs/out.wav'
CONTENT_FILENAME = sys.argv[1]
STYLE_FILENAME = sys.argv[2]

content_layer_ids = [0]
style_layer_ids = [0]

content_waveform, fs = read_audio(CONTENT_FILENAME)
content_spectrogram = trim(spectrum(content_waveform))

style_waveform, _ = read_audio(STYLE_FILENAME)
style_spectrogram = trim(spectrum(style_waveform))


def stack(layer):
    """
    Stack 3 2d-matrices
    :param layer: 2d-matrix
    :return: stacked matrix
    """
    return np.stack((layer, layer, layer), axis=2)


result = style_transfer(content_image=stack(content_spectrogram),
                                     style_image=stack(style_spectrogram),
                                     content_layer_ids=content_layer_ids,
                                     style_layer_ids=style_layer_ids,
                                     weight_content=10,
                                     weight_style=50,
                                     weight_denoise=0.0,
                                     num_iterations=120,
                                     step_size=0.5)

mixed_spectrogram = result[:, :, 0]

out_waveform = reconstruct(mixed_spectrogram, style_waveform.shape)
write_to_file(OUTPUT_FILENAME, out_waveform, fs)

spectrograms = {'content': content_spectrogram,
                'style': style_spectrogram,
                'out': mixed_spectrogram}

show(spectrograms)