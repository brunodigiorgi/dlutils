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
    "Basic": tf.contrib.rnn.BasicRNNCell,
    "BasicLSTM": tf.contrib.rnn.BasicLSTMCell,
    "GRU": tf.contrib.rnn.GRUCell,
    "LSTM": tf.contrib.rnn.LSTMCell,
}