import tensorflow as tf


TF_ACTIVATION = {
    "relu": tf.nn.relu,
    "sigmoid": tf.sigmoid,
    "tanh": tf.tanh
}

TF_OPTIMIZER = {
    "Adam": tf.train.AdamOptimizer,
    "Adagrad": tf.train.AdagradOptimizer,
    "RMSProp": tf.train.RMSPropOptimizer,
}

TF_RNN_CELL = {
    "Basic": tf.nn.rnn_cell.BasicRNNCell,
    "BasicLSTM": tf.nn.rnn_cell.BasicLSTMCell,
    "GRU": tf.nn.rnn_cell.GRUCell,
    "LSTM": tf.nn.rnn_cell.LSTMCell,
}