import tensorflow as tf
import numpy as np
from . import tf_constants


def rnn_stack(x, layers, keep_prob, scope=None, reuse=False):
    """
    Parameters
    ----------
    x : Tensor with shape [None (batch_size), num_steps, input_size]
    layers : list of dict
        [{"num_units": int, "cell_type": "Basic"|"BasicLSTM"|"GRU"|"LSTM"}, ...]
    keep_prob : unit Tensor

    Returns
    -------
    stack : tensorflow.python.ops.rnn_cell.MultiRNNCell
        The RNN Stack
    outputs : Tensor with shape [None (batch_size), layers[-1]['num_units']]
    """
    with tf.variable_scope(scope, reuse=reuse):
        cell_list = []
        for layer in layers:
            nunits = layer['num_units']
            cell_type = layer.get('cell_type', 'BasicLSTM')
            cell = tf_constants.TF_RNN_CELL[cell_type](nunits)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, keep_prob)
            cell_list.append(cell)
        stacked_cell = tf.nn.rnn_cell.MultiRNNCell(cell_list)

        initial_state = stacked_cell.zero_state(tf.shape(x)[0], dtype=tf.float32)
        x_ = tf.unpack(x, axis=1)  # create list of different time steps
        outputs, final_state = tf.nn.seq2seq.rnn_decoder(x_, initial_state, stacked_cell)
        outputs = tf.concat(1, outputs)

    return stacked_cell, outputs, initial_state, final_state


def dense_stack(x, layers, init_stddev=0.1, scope=None, reuse=False):
    """
    Parameters
    ----------
    x : Tensor with defined static shape [None (batch size), input_size]
    dense_layers : list of dict
        [{"num_units": int, "activation": "relu"|"sigmoid"|"tanh"}, ...]
        "activation" key is optional (no activation is used if not present)

    Returns
    -------
    net_vars : list of tensors
        a list of variables in the dense stack
    outputs : tensor with shape [None (batch_size), layers[-1]['num_units']]
    """
    in_size = x.get_shape()[1].value  # static shape
    out = x
    net_vars = []
    with tf.variable_scope(scope, reuse=reuse):
        for i, d in enumerate(layers):
            dense_w = tf.get_variable("dense" + str(i) + "_w", [in_size, d['num_units']])
            dense_b = tf.get_variable("dense" + str(i) + "_b", [d['num_units']])
            out = tf.matmul(out, dense_w) + dense_b
            if("activation" in d):
                out = tf_constants.TF_ACTIVATION[d["activation"]](out)
            in_size = d['num_units']
            net_vars.append(dense_w)
            net_vars.append(dense_b)
    return net_vars, out


def embedding(inputs, vocab_size, output_size, init_stddev=0.1):
    with tf.device("/cpu:0"):
        embedding = tf.Variable(tf.random_normal([vocab_size, output_size], stddev=init_stddev),
                                name="embedding", dtype=tf.float32)
        inputs = tf.nn.embedding_lookup(embedding, inputs)
    return inputs