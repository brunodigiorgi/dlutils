"""
References for variable length rnn:
* http://danijar.com/variable-sequence-lengths-in-tensorflow/
* http://r2rt.com/recurrent-neural-networks-in-tensorflow-iii-variable-length-sequences.html
* http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/
"""

import tensorflow as tf
import numpy as np
from . import tf_constants
from . import tf_base


def logits_sample(logits, temperature=1.0, avoid=[]):
    """
    helper function to sample an index from a logit array

    Parameters
    ----------
    temperature: float [0, 1]
        with 1 sample from the default softmax. With 0 choose the mode (max of out)
    avoid: list of symbols to avoid when sampling
    """

    logits = logits / temperature
    probs = np.exp(logits) / np.sum(np.exp(logits))

    # delete all avoid probs
    for i in avoid:
        probs[i] = 0
    probs = probs / np.sum(probs)  # re-normalize

    out = np.random.choice(len(probs), p=probs)
    return out


def sample(probs, temperature=1.0, avoid=[]):
    """
    helper function to sample an index from a probability array

    Parameters
    ----------
    temperature: float [0, 1]
        with 1 sample from the default softmax. With 0 choose the mode (max of out)
    avoid: list of symbols to avoid when sampling
    """

    # may produce underflow errors, call np.seterr(under='ignore') somewhere before this
    # for stability, avoid log(0)
    if(np.any(probs == 0)):
        k = np.min(probs[probs != 0]) * 0.01
        probs[probs == 0] = k

    logits = np.log(probs)
    return logits_sample(logits, temperature=temperature, avoid=avoid)


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


class RNNLM_TF_InputStage_OneHot:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.conf = {
            "name": "RNNLM_TF_InputStage_OneHot",
            "input_size": 1,
            "output_size": vocab_size,
        }

    def input_fn(self):
        inputs = tf.placeholder(tf.int32, [None, None, 1], name="inputs")
        seqlen = tf.placeholder(tf.int32, [None], name="seqlen")

        inputs_squeezed = tf.squeeze(inputs, [2])  # remove last dimension
        outputs = tf.one_hot(inputs_squeezed, self.vocab_size, dtype=tf.float32)
        return inputs, seqlen, outputs


class RNNLM_TF_InputStage_Regression:
    def __init__(self, input_size):
        self.input_size = input_size
        self.conf = {
            "name": "RNNLM_TF_InputStage_Regression",
            "input_size": input_size,
            "output_size": input_size,
        }

    def input_fn(self):
        inputs = tf.placeholder(tf.float32, [None, None, self.input_size], name="inputs")
        seqlen = tf.placeholder(tf.int32, [None], name="seqlen")

        outputs = inputs
        return inputs, seqlen, outputs


class RNNLM_TF_OutputStage_Classification:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.conf = {
            "name": "RNNLM_TF_OutputStage_Classification",
            "input_size": vocab_size,
            "output_size": vocab_size,
            "target_size": 1
        }

    def output_fn(self, logits):
        probs = tf.nn.softmax(logits)
        return probs

    def loss_fn(self, seqlen, logits):
        targets = tf.placeholder(tf.int32, [None, None, 1], name="targets")
        targets_flat = tf.reshape(targets, [-1])  # flatten

        seqlen_mask = tf.sequence_mask(seqlen)
        seqlen_mask_flat = tf.reshape(seqlen_mask, [-1])  # flatten
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets_flat)

        # boolean mask produces a UserWarning regarding the gradient
        # "Converting sparse IndexedSlices to a dense Tensor of unknown shape."
        # loss = tf.boolean_mask(loss, seqlen_mask_flat)
        # loss = tf.reduce_mean(loss)

        # equivalent but no Warning
        loss *= tf.cast(seqlen_mask_flat, dtype=tf.float32)
        loss = tf.reduce_sum(loss) / tf.cast(tf.reduce_sum(seqlen), dtype=tf.float32)

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

    def loss_fn(self, seqlen, outputs):
        targets = tf.placeholder(tf.float32, [None, None, self.output_size], name="targets")
        targets_flat = tf.reshape(targets, [-1])  # flatten

        seqlen_mask = tf.sequence_mask(seqlen)
        seqlen_mask_flat = tf.reshape(seqlen_mask, [-1])  # flatten

        outputs_flat = tf.reshape(outputs, [-1])  # flatten

        # mean square error
        loss = tf.square(outputs_flat - targets_flat)

        # boolean mask produces a UserWarning regarding the gradient
        # "Converting sparse IndexedSlices to a dense Tensor of unknown shape."
        # loss = tf.boolean_mask(loss, seqlen_mask_flat)
        # loss = tf.reduce_mean(loss)

        # equivalent but no Warning
        loss *= tf.cast(seqlen_mask_flat, dtype=tf.float32)
        loss = tf.reduce_sum(loss) / tf.cast(tf.reduce_sum(seqlen), dtype=tf.float32)

        return targets, loss


class RNNLM_TF():
    def __init__(self, input_stage, output_stage, feedback_stage, rnn_layers, dense_layers,
                 init_scale=.1, optimizer="Adagrad", learning_rate=.1, keep_prob=1., grad_clip=5.):
        """
        Creates a RNN architecture for language modeling with Tensorflow

        Parameters
        ----------
        input_stage: class
            implements
              (inputs, seqlen, outputs) = input_stage.input_fn()
            the function does not contain any trainable variable
        output_stage: class
            implements
              probs = output_stage.output_fn(outputs)
              targets, loss = output_stage.loss_fn(seqlen, outputs)
            both functions do not contain any trainable variable
        rnn_layer : see rnn_stack function
        dense_layer : see dense_stack function
        optimizer : key to tf_constant.TF_OPTIMIZER
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
        self.initialized = False

    def _create_model(self):
        tf.reset_default_graph()

        # input stage
        self.inputs, self.seqlen, outputs = self.input_stage.input_fn()

        # rnn
        self.keep_prob = tf.Variable(self.conf["keep_prob"], trainable=False, dtype=tf.float32)
        ret = tf_base.rnn_stack(outputs, self.seqlen, self.conf['rnn_layers'], self.keep_prob, scope='rnnlm')
        self.stacked_cell, outputs, self.initial_state, self.final_state = ret
        outputs = tf.reshape(outputs, [-1, self.conf['rnn_layers'][-1]['num_units']])

        # dense layers
        self.dense_vars, self.logits = tf_base.dense_stack(outputs, self.conf['dense_layers'], self.conf['init_scale'], scope='rnnlm')

        # output stage
        self.probs = self.output_stage.output_fn(self.logits)
        self.targets, self.loss = self.output_stage.loss_fn(self.seqlen, self.logits)

        # train
        self.lr = tf.Variable(self.conf["learning_rate"], trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.conf['grad_clip'])
        optimizer = tf_constants.TF_OPTIMIZER[self.conf['optimizer']](self.lr)
        self._train = optimizer.apply_gradients(zip(grads, tvars))
        self.need_reset_rnn_state = True  # initialize first rnn state

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
        # Add an Op to initialize global variables.
        init_op = tf.global_variables_initializer()
        self.session.run(init_op)
        self.initialized = True

    def train(self, inputs, targets, seqlen):
        self._set_keep_prob(self.conf['keep_prob'])
        feed = {self.inputs: inputs, self.targets: targets, self.seqlen: seqlen}

        if(self.need_reset_rnn_state):  # reset the initial state
            self.state = self.session.run(self.initial_state, feed)
            self.need_reset_rnn_state = False

        # rnn state feedback
        feed[self.initial_state] = self.state

        _, loss, self.state = self.session.run([self._train, self.loss, self.final_state], feed)
        return loss

    def test(self, inputs, targets, seqlen):
        self._set_keep_prob(1.)  # disable dropout: test is deterministic
        loss = self.session.run(self.loss, {self.inputs: inputs, self.targets: targets, self.seqlen: seqlen})
        return loss

    def _set_keep_prob(self, keep_prob):
        # set the keep_probability parameter of Dropout
        self.session.run(tf.assign(self.keep_prob, keep_prob))

    def save(self, save_path, global_step=None):
        return self.saver.save(self.session, save_path=save_path, global_step=global_step)

    def load(self, save_path):
        # save_path is tipically a value returned from a save()
        self.saver.restore(self.session, save_path=save_path)

    def generate(self, priming_seq, length):
        """
        generate a sequence using the learned language model

        Parameters
        ----------
        priming_seq: ndarray
            numpy array [T, I] where T is the number of priming time steps,
            and I is the input dimension
        length: int
            length of the generated sequence
        """
        self._set_keep_prob(1.)  # disable dropout: generation is deterministic

        if(priming_seq.shape[-1] != self.input_stage.conf["input_size"]):
            raise ValueError("incorrect shape of priming sequence")

        seqlen = np.array([priming_seq.shape[0]], dtype=np.int)
        inputs = np.expand_dims(priming_seq[:-1], axis=0)  # batch size = 1
        state = self.session.run(self.initial_state, {self.inputs: inputs})

        feed = {self.inputs: inputs, self.initial_state: state, self.seqlen: seqlen}
        state = self.session.run(self.final_state, feed)

        inputs = np.zeros((1, 1, self.input_stage.conf["input_size"]))

        out = []
        input_ = priming_seq[-1]
        seqlen[0] = 1
        for i in range(length):
            inputs[0, 0, :] = input_
            feed = {self.inputs: inputs, self.initial_state: state, self.seqlen: seqlen}
            [out_, state] = self.session.run([self.probs, self.final_state], feed)
            out_ = out_[0]  # batch_size = 1

            input_, out_ = self.feedback_stage(out_)
            assert((input_.size == self.input_stage.conf["input_size"]) and
                   (out_.size == self.output_stage.conf["target_size"]))

            out.append(out_)
        return np.array(out)

    def print_trainable(self):
        all_vars = tf.trainable_variables()
        for v in all_vars:
            print(v.name, v.get_shape())
