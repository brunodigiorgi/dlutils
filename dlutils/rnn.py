import tensorflow as tf
import numpy as np


TF_ACTIVATION = {
    "relu": tf.nn.relu,
    "sigmoid": tf.sigmoid,
    "tanh": tf.tanh
}

TF_OPTIMIZER = {
    "Adam": tf.train.AdamOptimizer,
    "Adagrad": tf.train.AdagradOptimizer,
}

TF_RNN_CELL = {
    "Basic": tf.nn.rnn_cell.BasicRNNCell,
    "BasicLSTM": tf.nn.rnn_cell.BasicLSTMCell,
    "GRU": tf.nn.rnn_cell.GRUCell,
    "LSTM": tf.nn.rnn_cell.LSTMCell,
}


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
            cell = TF_RNN_CELL[cell_type](nunits)
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
                out = TF_ACTIVATION[d["activation"]](out)
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


def sample(probs, temperature=1.0, avoid=[]):
    # helper function to sample an index from a probability array
    # may produce underflow errors, call np.seterr(under='ignore') somewhere before this

    # for stability, avoid log(0)
    if(np.any(probs == 0)):
        k = np.min(probs[probs != 0]) * 0.01
        probs[probs == 0] = k

    probs = np.log(probs) / temperature
    probs = np.exp(probs) / np.sum(np.exp(probs))

    # delete all avoid probs
    for i in avoid:
        probs[i] = 0
    probs = probs / np.sum(probs)  # re-normalize

    out = np.random.choice(len(probs), p=probs)
    return out


class RNNLM_TF_FeedbackStage_Sampler:
    def __init__(self, temperature=1.0, avoid=[]):
        self.temperature = temperature
        self.avoid = avoid
        self.conf = {
            "name": "RNNLM_TF_FeedbackNoop",
            "avoid": avoid,
            "temperature": temperature,
        }

    def __call__(self, out):
        np.seterr(under='ignore')  # for calling sample()
        out = sample(out, temperature=self.temperature, avoid=self.avoid)
        out = np.array((out,), dtype=np.int32)
        next_input = out
        return next_input, out


class RNNLM_TF_FeedbackStage_Noop:
    def __init__(self):
        self.conf = {"name": "RNNLM_TF_FeedbackNoop"}
        pass

    def __call__(self, out):
        return out, out


class RNNLM_TF_InputStage_Classification:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.conf = {
            "name": "RNNLM_TF_InputStage_Classification",
            "input_size": 1,
            "output_size": vocab_size,
        }

    def input_fn(self, num_steps):
        inputs = tf.placeholder(tf.int32, [None, num_steps, 1])
        inputs_squeezed = tf.squeeze(inputs, [2])  # remove last dimension
        inputs_squeezed = tf.reshape(inputs_squeezed, [-1, num_steps])  # assign static shape
        outputs = tf.one_hot(inputs_squeezed, self.vocab_size, dtype=tf.float32)
        return inputs, outputs


class RNNLM_TF_InputStage_Regression:
    def __init__(self, input_size):
        self.input_size = input_size
        self.conf = {
            "name": "RNNLM_TF_InputStage_Regression",
            "input_size": input_size,
            "output_size": input_size,
        }

    def input_fn(self, num_steps):
        inputs = tf.placeholder(tf.float32, [None, num_steps, self.input_size])
        outputs = inputs
        return inputs, outputs


class RNNLM_TF_OutputStage_Classification:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.conf = {
            "name": "RNNLM_TF_OutputStage_Classification",
            "input_size": vocab_size,
            "output_size": vocab_size,
            "target_size": 1
        }

    def output_fn(self, outputs):
        probs = tf.nn.softmax(outputs)
        return probs

    def loss_fn(self, num_steps, outputs):
        targets = tf.placeholder(tf.int32, [None, num_steps, 1])
        batch_size = tf.shape(targets)[0]  # dynamic
        targets_squeezed = tf.squeeze(targets, [2])  # remove last dimension
        targets_reshaped = tf.reshape(targets_squeezed, [-1])

        loss = tf.nn.seq2seq.sequence_loss_by_example([outputs],
                                                      [targets_reshaped],
                                                      [tf.ones_like(targets_reshaped, dtype=tf.float32)],
                                                      self.vocab_size)
        loss = tf.reduce_sum(loss) / tf.cast(batch_size, dtype=tf.float32) / num_steps
        return targets, loss


class RNNLM_TF_OutputStage_Regression:
    def __init__(self, output_size):
        self.output_size = output_size
        self.conf = {
            "name": "RNNLM_TF_OutputStage_Regression",
            "input_size": output_size,
            "output_size": output_size,
            "target_size": output_size
        }

    def output_fn(self, outputs):
        return outputs

    def loss_fn(self, num_steps, outputs):
        targets = tf.placeholder(tf.float32, [None, num_steps, self.output_size])
        targets_reshaped = tf.reshape(targets, tf.shape(outputs))

        # mean square error
        loss = tf.reduce_mean(tf.square(outputs - targets_reshaped))
        return targets, loss


class RNNLM_TF():
    def __init__(self, input_stage, output_stage, feedback_stage, num_steps, rnn_layers, dense_layers,
                 init_scale=.1, optimizer="Adagrad", learning_rate=.1, keep_prob=1., grad_clip=5.):
        """
        Creates a RNN architecture for language modeling with Tensorflow

        Parameters
        ----------
        input_stage: class
            implements
              (inputs, outputs) = input_stage.input_fn(num_steps)
            the function does not contain any trainable variable
        output_stage: class
            implements
              probs = output_stage.output_fn(outputs)
              targets, loss = output_stage.loss_fn(num_steps, outputs)
            both functions do not contain any trainable variable
        num_steps : int
        rnn_layer : see rnn_stack function
        dense_layer : see dense_stack function
        optimizer : "Adam" | "Adagrad"
        learning_rate : float
        keep_prob : float [0, 1]
        grad_clip : float
        """

        self.input_stage = input_stage
        self.output_stage = output_stage
        self.feedback_stage = feedback_stage

        self.conf = {
            "input_stage": input_stage.conf,
            "output_stage": output_stage.conf,
            "feedback_stage": feedback_stage.conf,
            "num_steps": num_steps,
            "rnn_layers": rnn_layers,
            "dense_layers": dense_layers,
            "init_scale": init_scale,
            "learning_rate": learning_rate,
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

        # input stage
        self.inputs, outputs = self.input_stage.input_fn(self.conf['num_steps'])

        # rnn
        self.keep_prob = tf.Variable(self.conf["keep_prob"], trainable=False, dtype=tf.float32)
        ret = rnn_stack(outputs, self.conf['rnn_layers'], self.keep_prob, scope='rnnlm')
        self.stacked_cell, outputs, self.initial_state, self.final_state = ret
        outputs = tf.reshape(outputs, [-1, self.conf['rnn_layers'][-1]['num_units']])

        # dense layers
        self.dense_vars, outputs = dense_stack(outputs, self.conf['dense_layers'], self.conf['init_scale'], scope='rnnlm')

        # output stage
        self.probs = self.output_stage.output_fn(outputs)
        self.targets, self.loss = self.output_stage.loss_fn(self.conf['num_steps'], outputs)

        # train
        self.lr = tf.Variable(self.conf["learning_rate"], trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.conf['grad_clip'])
        optimizer = TF_OPTIMIZER[self.conf['optimizer']](self.lr)
        self._train = optimizer.apply_gradients(zip(grads, tvars))
        self.need_reset_rnn_state = True  # initialize rnn state

        # generative graph, needed to use inputs with num_steps = 1
        self.g_inputs, g_outputs = self.input_stage.input_fn(1)
        ret = rnn_stack(g_outputs, self.conf['rnn_layers'], self.keep_prob, scope='rnnlm', reuse=True)
        _, g_outputs, self.g_initial_state, self.g_final_state = ret
        _, g_outputs = dense_stack(g_outputs, self.conf['dense_layers'], self.conf['init_scale'], scope='rnnlm', reuse=True)
        self.g_outputs = self.output_stage.output_fn(g_outputs)

        # saver
        self.saver = tf.train.Saver(max_to_keep=5)  # keep only 5 most recent checkpoints

    def _check_conf(self):
        if(len(self.conf['dense_layers']) > 0):
            assert(self.conf['dense_layers'][-1]['num_units'] == self.output_stage.conf["input_size"])
        else:
            assert(self.conf['rnn_layers'][-1]['num_units'] == self.output_stage.conf["input_size"])
        assert((self.conf['keep_prob'] >= 0.) and (self.conf['keep_prob'] <= 1.))

    def set_learning_rate(self, lr):
        self.session.run(tf.assign(self.lr, lr))

    def new_sequence(self):
        """
        To be called before train() when starting a new sequence
        Make sure that, for the next training batch, rnn state will be re-initialized
        """
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

        # rnn state feedback
        feed[self.initial_state] = self.state

        # for s, val in zip(traverse(self.initial_state), traverse(self.state)):
        #     feed[s] = val

        # feedback rnn state
        # for i, (c, h) in enumerate(self.initial_state):
        #     feed[c] = self.state[i].c  # cell state
        #     feed[h] = self.state[i].h  # hidden state

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

    def generate(self, priming_seq, length, temperature=1.0, avoid=[]):
        self._set_keep_prob(1.)
        state = self.session.run(self.stacked_cell.zero_state(1, tf.float32))
        x = np.zeros((1, 1, self.input_stage.conf["input_size"]))

        for input_ in priming_seq[:-1]:
            x[0, 0, :] = input_
            feed = {self.g_inputs: x, self.g_initial_state: state}
            [state] = self.session.run([self.g_final_state], feed)

        out = []
        input_ = priming_seq[-1]
        for i in range(length):
            x[0, 0, :] = input_
            feed = {self.g_inputs: x, self.g_initial_state: state}
            [out_, state] = self.session.run([self.g_outputs, self.g_final_state], feed)
            out_ = out_[0]  # batch_size = 1

            input_, out_ = self.feedback_stage(out_)
            assert((input_.size == self.input_stage.conf["input_size"]) and
                   (out_.size == self.output_stage.conf["target_size"]))

            out.append(out_)
        return out

    def print_trainable(self):
        all_vars = tf.trainable_variables()
        for v in all_vars:
            print(v.name, v.get_shape())
