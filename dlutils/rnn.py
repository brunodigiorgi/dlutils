import tensorflow as tf


TF_activation = {
    "relu": tf.nn.relu,
    "sigmoid": tf.sigmoid,
    "tanh": tf.tanh
}


def lstm_stack(x, layers):
    """
    Parameters
    ----------
    x : Tensor with shape [None (batch_size), num_steps, input_size]
    layers : list of dict
        [{"num_units": int}, ...]

    Returns
    -------
    stack : tensorflow.python.ops.rnn_cell.MultiRNNCell
        The RNN Stack
    outputs : Tensor with shape [None (batch_size), layers[-1]['num_units']]
    """
    x_ = tf.unpack(x, axis=1)  # create list of different time steps
    lstm_cells = []
    for layer in layers:
        lstm = tf.nn.rnn_cell.BasicLSTMCell(layer['num_units'], forget_bias=0.0)
        lstm_cells.append(lstm)
    stacked_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)
    _initial_state = stacked_cell.zero_state(tf.shape(x)[0], dtype=tf.float32)
    outputs, state = tf.nn.rnn(stacked_cell, x_, initial_state=_initial_state)
    outputs = outputs[-1]  # consider just the last output
    outputs.set_shape(tf.TensorShape([None, layers[-1]['num_units']]))
    return stacked_cell, outputs


def dense_stack(x, layers, init_stddev=0.1):
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
    net_vars = []
    for i, d in enumerate(layers):
        dense_w = tf.Variable(tf.random_normal([in_size, d['num_units']], stddev=init_stddev),
                              name="dense" + str(i) + "_w", dtype=tf.float32)
        dense_b = tf.Variable(tf.random_normal([d['num_units']], stddev=init_stddev),
                              name="dense" + str(i) + "_b", dtype=tf.float32)
        x = tf.matmul(x, dense_w) + dense_b
        if("activation" in d):
            x = TF_activation[d["activation"]](x)
        in_size = d['num_units']
        net_vars.append(dense_w)
        net_vars.append(dense_b)
    return net_vars, x


class RNNTensorFlow():
    def __init__(self, num_steps, rnn_layers, dense_layers, input_size, output_size, init_scale, learning_rate, output_type='cls'):
        """
        Parameters
        ----------
        rnn_layers : list of dict
            [{"num_units": int}, ...]
        dense_layers : list of dict
            [{"num_units": int, "activation": "relu"|"sigmoid"|"tanh"}, ...]
            "activation" key is optional
        output_type : string
            'cls'|'reg'
        """
        if(len(dense_layers) > 0):
            assert(dense_layers[-1]['num_units'] == output_size)
        else:
            assert(rnn_layers[-1]['num_units'] == output_size)

        self.conf = {
            "num_steps": num_steps,
            "rnn_layers": rnn_layers,
            "dense_layers": dense_layers,
            "input_size": input_size,
            "output_size": output_size,
            "init_scale": init_scale,
            "learning_rate": learning_rate,
            "output_type": output_type,
        }

        self._create_model()

        # the session might eventually be externalized from the model, I don't see the need to do it now
        self.session = tf.Session()

    def _create_model(self):
        tf.reset_default_graph()
        self.inputs = tf.placeholder(tf.float32, shape=[None, self.conf['num_steps'], self.conf['input_size']])
        self.targets = tf.placeholder(tf.float32, shape=[None, self.conf['output_size']])

        # rnn layers
        self.stacked_cell, outputs = lstm_stack(self.inputs, self.conf['rnn_layers'])

        # dense layers
        if(len(self.conf['dense_layers']) > 0):
            self.dense_vars, outputs = dense_stack(outputs, self.conf['dense_layers'], self.conf['init_scale'])

        if(self.conf['output_type'] == 'cls'):
            self.probs = tf.nn.softmax(outputs)
            self.prediction = self.probs
            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(outputs, self.targets))
            self.loss = self.cross_entropy
        elif(self.conf['output_type'] == 'reg'):
            self.prediction = outputs
            self.mse = tf.reduce_mean(tf.square(outputs - self.targets))
            self.loss = self.mse

        optimizer = tf.train.AdagradOptimizer(learning_rate=self.conf['learning_rate'])
        self._train = optimizer.minimize(self.loss)
        self.saver = tf.train.Saver(max_to_keep=5)  # keep only 5 most recent checkpoints

    def initialize(self):
        init = tf.initialize_all_variables()
        self.session.run(init)

    def train(self, inputs, targets):
        _, loss = self.session.run([self._train, self.loss], {self.inputs: inputs, self.targets: targets})
        return loss

    def test(self, inputs, targets):
        loss = self.session.run(self.loss, {self.inputs: inputs, self.targets: targets})
        return loss

    def predict(self, inputs):
        prediction = self.session.run(self.prediction, {self.inputs: inputs})
        return prediction

    def save(self, fn):
        self.saver.save(self.session, fn)

    def load(self, fn):
        self.saver.restore(self.session, fn)

    def print_trainable(self):
        all_vars = tf.trainable_variables()
        for v in all_vars:
            print(v.name, v.get_shape())

    def __del__(self):
        self.session.close()
