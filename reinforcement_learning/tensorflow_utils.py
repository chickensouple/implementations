"""
General Utilities for Tensorflow
"""
import numpy as np
import tensorflow as tf

def fully_connected(input_var, num_output, activation=None, use_bias=True, var_init=None, batch_norm=False, **kwargs):
    """
    Creates a fully connected neural net layer. the layer will be given as sigma(x*W + b), where W are the weights, b
    are the biases, and sigma is an activation function

    This will generate the variables for weights and biases
    
    To prevent naming conflicts, this should be wrapped in a tf.variable_scope()
    Example usage:

    inputs = tf.placeholder(tf.float32, [None, N], name='state')
    with tf.variable_scope('hidden'):
        hidden = fully_connected(inputs, 400, activation=tf.nn.relu)
    with tf.variable_scope('out'):
        output = fully_connected(hidden, 1)

    Args:
        input_var (tf tensor): the input variable into this layer. Input variable should be of shape [None, N]
        num_output (int): the number of neurons and outputs in this layer
        activation (tf op, optional): activation function for the layer. If None is given there will be no nonlinearity
        use_bias (bool, optional): include a bias term in the fully connected layer. Default is to include a bias term
        var_init (func, optional): initializer for the weights. If None is given, Default is to use Xavier Initialization
        batch_norm (bool, optional): if True, use batch normalization. NOT IMPLEMENTED YET
        **kwargs: extra arguments to pass into tf.get_variable() 
    
    Returns:
        tf tensor: reference to output of the layer
    """
    num_input = int(input_var.shape[1])

    # initialize with xavier initialization if no initialization is given
    if var_init == None:
        var_init = tf.contrib.layers.xavier_initializer()
    init_lambd = lambda shape, dtype, partition_info=None: var_init(shape, dtype=dtype)

    w = tf.get_variable(name='w', shape=[num_input, num_output], initializer=init_lambd, **kwargs)
    affline_op = tf.matmul(input_var, w)

    # add bias term if needed
    if use_bias:
        b = tf.get_variable(name='b', shape=[1, num_output], initializer=init_lambd, **kwargs)
        affline_op = tf.add(affline_op, b)

    # add batch norm if needed
    # TODO: not functional yet
    if batch_norm:
        act_input = tf.nn.batch_normalization(inputs=affline_op, training=True)
    else:
        act_input = affline_op

    # set activation function (if any)
    if activation == None:
        output = act_input
    else:
        output = activation(act_input)

    return output