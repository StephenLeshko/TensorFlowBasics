import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display



model = models.load_model('audioBasic1.h5')

#commmands
commands = ['down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes']


#turns audio into tensor
def decode_audio(audio_binary):
    # Decode WAV-encoded audio files to `float32` tensors, normalized
    # to the [-1.0, 1.0] range. Return `float32` audio and a sample rate.
    audio, _ = tf.audio.decode_wav(contents=audio_binary)
    # Since all the data is single channel (mono), drop the `channels`
    # axis from the array.
    return tf.squeeze(audio, axis=-1)

#turns wave file into spectrogram
def get_spectrogram(waveform):
    # Zero-padding for an audio waveform with less than 16,000 samples.
    input_len = 16000
    waveform = waveform[:input_len]
    zero_padding = tf.zeros(
        [16000] - tf.shape(waveform),
        dtype=tf.float32)
    # Cast the waveform tensors' dtype to float32.
    waveform = tf.cast(waveform, dtype=tf.float32)
    # Concatenate the waveform with `zero_padding`, which ensures all audio
    # clips are of the same length.
    equal_length = tf.concat([waveform, zero_padding], 0)
    # Convert the waveform to a spectrogram via a STFT.
    spectrogram = tf.signal.stft(
        equal_length, frame_length=255, frame_step=128)
    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)
    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram

#gets the file

def predict_audio(file_name):
    audio_binary = tf.io.read_file(file_name)
    waveform = decode_audio(audio_binary)
    spectrogram = get_spectrogram(waveform)
    spectrogram = tf.reshape(spectrogram, [1, 124, 129, 1])
    prediction = model(spectrogram)
    return commands[int(tf.argmax(prediction, 1).numpy())]



sample_file = 'test.wav'

audio_binary = tf.io.read_file(sample_file)

waveform = decode_audio(audio_binary)

print('decoded\n\n')

spec = get_spectrogram(waveform)
spec = tf.reshape(spec, [1, 124, 129, 1])


prediction = model(spec)

#gets you the predicted value stored in a variable
print('prediction', commands[int(tf.argmax(prediction, 1).numpy())])

print('test', predict_audio('test.wav'))
print('left', predict_audio('left.wav'))
print('guess', predict_audio('guess.wav'))





# plt.bar(commands, tf.nn.softmax(prediction[0]))
# plt.title(f'Predictions for "{commands[0]}"')
# plt.show()

