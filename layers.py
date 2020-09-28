import math
import numpy as np
import tensorflow as tf


def conv2d(input_, output_dim, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2d", reuse=False, padding='SAME'):
    with tf.variable_scope(name, reuse=reuse):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.contrib.layers.xavier_initializer())
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)

        biases = tf.get_variable('biases', [output_dim],
                                 initializer=tf.constant_initializer(0.0))

        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

    return conv

def batch_norm(inputs, name=None, train=True, reuse=False):
    return tf.contrib.layers.batch_norm(inputs=inputs,is_training=train,
                                      reuse=reuse,scope=name,scale=True)


def relu(x):
    return tf.nn.relu(x)

def MaxPooling(x, k, stride=None, padding = 'SAME', name=None):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding= padding, name=name)


def Dense(input_tensor, num_inputs, num_outputs, name=None, reuse=False):
    """
    Handy wrapper function for convolutional networks.
    Performs an affine layer (fully-connected) on the input tensor.
    """
    shape = [num_inputs, num_outputs]

    # initialize weights and biases of the affine layer
    with tf.variable_scope(name, reuse=reuse):
        W = tf.get_variable('W', shape, initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('b', shape=shape[-1], initializer=tf.zeros_initializer)

        fc = tf.matmul(input_tensor, W) + b

    return fc


def Flatten(layer):
    """Handy function for flattening the result of a conv2D or
    maxpool2D to be used for a fully-connected (affine) layer.
    """
    layer_shape = layer.get_shape()
    # num_features = tf.reduce_prod(tf.shape(layer)[1:])
    num_features = layer_shape[1:].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])

    return layer_flat, num_features


def Global_avg_pool(x, padding='SAME', name=None):

    #layer_shape = x.get_shape()
    shape = x.get_shape()
    #num_features = layer_shape[1:].num_elements()
    #layer_flat = tf.reshape(x, [-1, num_features])

    #weight= 1/(layer_shape[1:3].num_elements())

    #GAP= tf.matmul(layer_flat, tf.constant(weight,dtype=tf.float32, shape=[num_features,layer_shape[3]]))

    return tf.nn.avg_pool(x, ksize=[1, shape[1], shape[2], 1], strides=[1, shape[1], shape[2], 1],
                          padding=padding, name=name)


def Global_max_pool(x, padding='SAME', name=None):
    shape = x.get_shape()

    return tf.nn.max_pool(x, ksize=[1, shape[1], shape[2], 1], strides=[1, shape[1], shape[2], 1],
                          padding=padding, name=name)


