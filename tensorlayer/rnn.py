
import tensorflow as tf
import tensorlayer as tl
from tensorflow.examples.tutorials.mnist import input_data


batch_size = 100
num_steps = 50
vocab_size = 100
hidden_size = 1000
init_scale = 100
keep_prob = 0.8


mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
network = tl.layers.EmbeddingInputlayer(inputs=mnist,
                                        vocabulary_size=vocab_size,
                                        embedding_size=hidden_size,
                                        E_init=tf.random_uniform_initializer(-init_scale, init_scale),
                                        name='embedding_layer')

network = tl.layers.DropoutLayer(network, keep=keep_prob, name='drop1')

network = tl.layers.RNNLayer(network,
                             cell_fn=tf.nn.rnn_cell.BasicLSTMCell,
                             cell_init_args={'forget_bias': 0.0},# 'state_is_tuple': True},
                             n_hidden=hidden_size,
                             initializer=tf.random_uniform_initializer(-init_scale, init_scale),
                             n_steps=num_steps,
                             return_last=False,
                             name='basic_lstm_layer1')