import time
import sys

import tensorflow as tf

from audio_utils import *

# set up a graph to find feature activations
# filters are pre-set constants
FILTER_WIDTH = 10
FILTER_HEIGHT = 10
FILTER_IN_CHANNEL = 1
FILTER_OUT_CHANNEL = 10

# optimization iterations
MAX_ITER = 30

# style weight, with respect to content weight, in the mixed signal
CONTENT_WEIGHT = 1
STYLE_WEIGHT = 0.7


# total loss
loss = []


def format_input(input):
    """
    Convert a 2D np.array representation of spectrogram to a 4D volume
    so that it can be used as input to the CNN
    :param input: a 2D np array
    :return: a 4D np array
    """
    input_3d = np.expand_dims(input, axis=2)
    return np.expand_dims(input_3d, axis=3)


def format_output(eval):
    """
    Squash a 4D tensor eval result into a 2D array so that it can be used as
    a representation of spectrogram
    :param output:
    :return:
    """
    return eval[:, :, 0, 0]

content_audio, fs = read_audio(sys.argv[1])
content_img = get_spectrum(content_audio)

style_audio, _ = read_audio(sys.argv[2])
style_img = get_spectrum(style_audio)

def gram(tensor):
    shape = tensor.get_shape()

    # Get the number of feature channels for the input tensor,
    # which is assumed to be from a convolutional layer with 4-dim.
    num_channels = int(shape[3])

    # Reshape the tensor so it is a 2-dim matrix. This essentially
    # flattens the contents of each feature-channel.
    matrix = tf.reshape(tensor, shape=[-1, num_channels])

    # Calculate the Gram-matrix as the matrix-product of
    # the 2-dim matrix with itself. This calculates the
    # dot-products of all combinations of the feature-channels.
    return tf.matmul(tf.transpose(matrix), matrix)


def log_loss(loss_eval):
  print('loss = ' + str(loss_eval))
  loss.append(loss_eval)


filter_val = np.random.rand(FILTER_HEIGHT,
                                FILTER_WIDTH,
                                FILTER_IN_CHANNEL,
                                FILTER_OUT_CHANNEL).astype(np.float32)

# get the activation of content and style inputs on the CNN
with tf.Graph().as_default():
    input = tf.placeholder('float32')
    filters = tf.constant(filter_val)

    conv = tf.nn.conv2d(input,
                        filters,
                        [1, 1, 1, 1],
                        padding='SAME',
                        name='conv')
    relu = tf.nn.relu(conv,
                      name='relu')

    gram_mat = gram(relu)

    with tf.Session() as sess:
        content_features = sess.run(relu, feed_dict={input: format_input(content_img)})
        style_features = sess.run(relu, feed_dict={input: format_input(style_img)})

with tf.Graph().as_default():
    # The mixed-image is initialized with random noise.
    # It is the same size as the content-image.
    mixed_img = np.random.rand(*content_img.shape).astype(np.float32)
    input = tf.Variable(format_input(mixed_img))

    filters = tf.constant(filter_val)

    conv = tf.nn.conv2d(input,
                        filters,
                        [1, 1, 1, 1],
                        padding='SAME',
                        name='conv')
    relu = tf.nn.relu(conv,
                      name='relu')

    gram_mat = gram(relu)

    content_features_tensor = tf.constant(content_features)
    style_features_tensor = tf.constant(style_features)

    content_loss = tf.nn.l2_loss(relu - content_features_tensor)
    style_loss = tf.nn.l2_loss(relu - style_features_tensor)

    total_loss = CONTENT_WEIGHT * content_loss + STYLE_WEIGHT * style_loss

    optimizer = tf.contrib.opt.ScipyOptimizerInterface(
        total_loss, method='L-BFGS-B', options={'maxiter': MAX_ITER})

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        optimizer.minimize(sess,
                           loss_callback=log_loss,
                           fetches=[total_loss])

        mixed_img = format_output(input.eval())


plt.plot(loss)
plt.xlabel('iterations')
plt.ylabel('loss')
plt.show()


out_waveform = reconstruct(mixed_img, content_audio.shape)
write_to_file('outputs/out_{}.wav'.format(int(time.time())), out_waveform, fs)


spectrograms = {'content': content_img,
                'style': style_img,
                'out': mixed_img}

show(spectrograms)