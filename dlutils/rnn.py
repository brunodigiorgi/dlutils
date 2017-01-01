import tensorflow as tf
import numpy as np


TF_activation = {
    "relu": tf.nn.relu,
    "sigmoid": tf.sigmoid,
    "tanh": tf.tanh
}

TF_Optimizer = {
    "Adam": tf.train.AdamOptimizer,
    "Adagrad": tf.train.AdagradOptimizer,
}


def rnn_stack(x, layers, keep_prob, scope=None, reuse=False):
    """
    Parameters
    ----------
    x : Tensor with shape [None (batch_size), num_steps, input_size]
    layers : list of dict
        [{"num_units": int}, ...]
    keep_prob : unit Tensor        

    Returns
    -------
    stack : tensorflow.python.ops.rnn_cell.MultiRNNCell
        The RNN Stack
    outputs : Tensor with shape [None (batch_size), layers[-1]['num_units']]
    """
    with tf.variable_scope(scope, reuse=reuse):
        lstm_cells = []
        for layer in layers:
            lstm = tf.nn.rnn_cell.BasicLSTMCell(layer['num_units'])
            lstm = tf.nn.rnn_cell.DropoutWrapper(lstm, keep_prob)
            lstm_cells.append(lstm)
        stacked_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)

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
                out = TF_activation[d["activation"]](out)
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


def sample(probs, temperature=1.0):
    # helper function to sample an index from a probability array
    # may produce underflow errors, call np.seterr(under='ignore') somewhere before this
    probs = np.log(probs) / temperature
    probs = np.exp(probs) / np.sum(np.exp(probs))
    return np.random.choice(len(probs), p=probs)  # more stable than np.argmax(np.random.multinomial(1, a, 1))


class RNNLM_Tensorflow():
    def __init__(self, num_steps, rnn_layers, dense_layers, vocab_size,
                 init_scale=.1, optimizer="Adagrad", learning_rate=.1, keep_prob=1., decay_rate=1.0, grad_clip=5.):

        self.conf = {
            "num_steps": num_steps,
            "rnn_layers": rnn_layers,
            "dense_layers": dense_layers,
            "vocab_size": vocab_size,
            "init_scale": init_scale,
            "learning_rate": learning_rate,
            "decay_rate": decay_rate,
            "grad_clip": grad_clip,
            "keep_prob": keep_prob,
            "optimizer": optimizer,
        }

        self._check_conf()
        self._create_model()

        # the session might eventually be externalized from the model, I don't see the need to do it now
        self.session = tf.Session()

    def _create_model(self):
        tf.reset_default_graph()

        self.inputs = tf.placeholder(tf.int32, [None, self.conf['num_steps'], 1])
        self.targets = tf.placeholder(tf.int32, [None, self.conf['num_steps'], 1])
        batch_size = tf.shape(self.inputs)[0]  # dynamic

        # preprocess
        inputs_squeezed = tf.squeeze(self.inputs, [2])  # remove last dimension
        targets_squeezed = tf.squeeze(self.targets, [2])  # remove last dimension
        inputs_squeezed = tf.reshape(inputs_squeezed, [-1, self.conf['num_steps']])  # assign static shape
        outputs = tf.one_hot(inputs_squeezed, self.conf['vocab_size'], dtype=tf.float32)

        # rnn
        self.keep_prob = tf.Variable(self.conf["keep_prob"], trainable=False, dtype=tf.float32)
        self.stacked_cell, outputs, self.initial_state, self.final_state = rnn_stack(outputs, self.conf['rnn_layers'], self.keep_prob, scope='rnnlm')
        outputs = tf.reshape(outputs, [-1, self.conf['rnn_layers'][-1]['num_units']])

        # dense layers
        self.dense_vars, outputs = dense_stack(outputs, self.conf['dense_layers'], self.conf['init_scale'], scope='rnnlm')

        # output
        self.logits = outputs
        self.probs = tf.nn.softmax(self.logits)
        self.prediction = self.probs

        # loss
        targets_reshaped = tf.reshape(targets_squeezed, [-1])
        loss = tf.nn.seq2seq.sequence_loss_by_example([self.logits],
                                                      [targets_reshaped],
                                                      [tf.ones_like(targets_reshaped, dtype=tf.float32)],
                                                      self.conf['vocab_size'])
        self.cost = tf.reduce_sum(loss) / tf.cast(batch_size, dtype=tf.float32) / self.conf['num_steps']
        self.loss = self.cost

        # train
        self.lr = tf.Variable(self.conf["learning_rate"], trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), self.conf['grad_clip'])
        optimizer = TF_Optimizer[self.conf['optimizer']](self.lr)
        self._train = optimizer.apply_gradients(zip(grads, tvars))
        self.need_reset_rnn_state = True  # initialize rnn state

        # generative graph
        self.g_inputs = tf.placeholder(tf.int32, [None, 1, 1])
        g_inputs_squeezed = tf.squeeze(self.g_inputs, [2])  # remove last dim
        g_outputs = tf.one_hot(g_inputs_squeezed, self.conf['vocab_size'], dtype=tf.float32)
        _, g_outputs, self.g_initial_state, self.g_final_state = rnn_stack(g_outputs, self.conf['rnn_layers'], self.keep_prob, scope='rnnlm', reuse=True)
        _, g_outputs = dense_stack(g_outputs, self.conf['dense_layers'], self.conf['init_scale'], scope='rnnlm', reuse=True)
        self.g_logits = g_outputs
        self.g_probs = tf.nn.softmax(self.g_logits)

        # saver
        self.saver = tf.train.Saver(max_to_keep=5)  # keep only 5 most recent checkpoints

    def _check_conf(self):
        if(len(self.conf['dense_layers']) > 0):
            assert(self.conf['dense_layers'][-1]['num_units'] == self.conf['vocab_size'])
        else:
            assert(self.conf['rnn_layers'][-1]['num_units'] == self.conf['vocab_size'])
        assert((self.conf['keep_prob'] >= 0.) and (self.conf['keep_prob'] <= 1.))

    def set_epoch(self, e):
        self.session.run(tf.assign(self.lr, self.conf["learning_rate"] * (self.conf["decay_rate"] ** e)))

    def new_sequence(self):
        self.need_reset_rnn_state = True

    def initialize(self):
        init = tf.initialize_all_variables()
        self.session.run(init)

    def train(self, inputs, targets):
        self._set_keep_prob(self.conf['keep_prob'])
        feed = {self.inputs: inputs, self.targets: targets}

        if(self.need_reset_rnn_state):
            self.state = self.session.run(self.initial_state, feed)
            self.need_reset_rnn_state = False

        # feedback rnn state
        for i, (c, h) in enumerate(self.initial_state):
            feed[c] = self.state[i].c  # cell state
            feed[h] = self.state[i].h  # hidden state

        _, loss, self.state = self.session.run([self._train, self.loss, self.final_state], feed)
        return loss

    def test(self, inputs, targets):
        self._set_keep_prob(1.)
        loss = self.session.run(self.loss, {self.inputs: inputs, self.targets: targets})
        return loss

    def _set_keep_prob(self, keep_prob):
        self.session.run(tf.assign(self.keep_prob, keep_prob))

    def save(self, fn):
        self.saver.save(self.session, fn)

    def load(self, fn):
        self.saver.restore(self.session, fn)

    def generate(self, priming_seq, length, temperature=1.0):
        self._set_keep_prob(1.)
        state = self.session.run(self.stacked_cell.zero_state(1, tf.float32))
        x = np.zeros((1, 1, 1))
        np.seterr(under='ignore')  # for calling sample()

        for sym in priming_seq[:-1]:
            x[0, 0, 0] = sym
            feed = {self.g_inputs: x, self.g_initial_state: state}
            [state] = self.session.run([self.g_final_state], feed)

        out = []
        sym = priming_seq[-1]
        for i in range(length):
            x[0, 0, 0] = sym
            feed = {self.g_inputs: x, self.g_initial_state: state}
            [probs, state] = self.session.run([self.g_probs, self.g_final_state], feed)
            probs = probs[0]  # batch_size = 1
            sym = sample(probs, temperature=temperature)
            out.append(sym)

        return out

    def print_trainable(self):
        all_vars = tf.trainable_variables()
        for v in all_vars:
            print(v.name, v.get_shape())