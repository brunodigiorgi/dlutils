import numpy as np
import random
import collections
import math
from .utils import frame_ndarray


def format_sequence(in_seq, target_seq, batch_size, num_steps, overlap=.5):
    """
    Split a sequence into a list of batches.
    Each batch is a list of two ndarray [x, y]
    Each ndarray has dim=[batch_size, num_steps, input_size]
    """
    assert(len(in_seq) == len(target_seq))
    assert(len(in_seq) > num_steps)
    data_len = len(in_seq)
    batch_len = data_len // batch_size
    hop_size = batch_len
    last_batch = batch_size - 1

    if(overlap is not None):
        batch_len = - data_len / (overlap * last_batch - last_batch - 1)
        hop_size = batch_len * (1 - overlap)
        hop_size = math.floor(hop_size)
        batch_len = data_len - hop_size * last_batch

    # make sure that hop_size is not a multiple of num_steps (for variety in batches)
    # Effectiveness of this step is not yet proved
    if((hop_size % num_steps == 0) and (hop_size > 1)):
        hop_size -= 1
        batch_len = data_len - hop_size * last_batch

    in_seq_reshaped = frame_ndarray(in_seq[0:(hop_size * last_batch + batch_len)], batch_len, hop_size)
    target_seq_reshaped = frame_ndarray(target_seq[0:(hop_size * last_batch + batch_len)], batch_len, hop_size)
    nbatches = batch_len // num_steps
    out = []
    for i in range(nbatches):
        x_ = in_seq_reshaped[:, i * num_steps:(i + 1) * num_steps, ...]
        y_ = target_seq_reshaped[:, i * num_steps:(i + 1) * num_steps, ...]
        out.append([x_, y_])
    return out


class Dataset_seq2seq():
    def __init__(self, in_data, batch_size, num_steps, target_data=None, name='', seq_processor=None):
        """
        Parameters
        ----------
        data : list of numpy ndarray
            if string is a npz file containing a list of ndarray [seq_length x frame_size]
        dataset_transformation : callable
            applied to every original or augmented sequence
        transformation_mode : 'pre' | 'post'
            select if the transformation is applied before or after the augmentations
        dataset_augmentations : list of callable
            create a new sequence from a given one
        """
        self.data = in_data
        self.target_data = target_data
        self.name = name
        self.seq_processor = seq_processor

        if(self.data[0].ndim == 1):
            self.input_size = 1
            for i, d in enumerate(self.data):
                self.data[i] = self.data[i].reshape(-1, 1)
        else:
            self.input_size = self.data[0].shape[1]

        self.batch_size = batch_size
        self.num_steps = num_steps

        self.nseq = len(self.data)
        self.slen = [len(s) for s in self.data]

        # parse target_data if present
        if(self.target_data is not None):
            assert(len(self.target_data) == len(self.data))
            for i in range(self.nseq):
                # assume same length
                assert(self.target_data[i].size == self.data[i].size)
                if(self.target_data[i].ndim == 1):
                    self.target_data[i] = self.target_data[i].reshape(-1, 1)

        self.conf = {
            "name": self.name,
            "batch_size": int(self.batch_size),
            "num_steps": int(self.num_steps),
            "nseq": self.nseq,
            "slen": self.slen,
            "tot_len": sum(self.slen),
            "max_len": max(self.slen),
            "min_len": min(self.slen),
        }

    def format(self, iseq):
        seq = self.data[iseq]

        # data is a tuple input_seq, target_seq
        data = (seq[:-1], seq[1:])
        if(self.target_data is not None):
            data = (seq, self.target_data[iseq])

        # seqs is the list containing data + all augmented sequences
        seqs = [data]
        if(self.seq_processor is not None):
            seqs.extend(self.seq_processor(data))

        out = []
        for s in seqs:
            out.append(format_sequence(s[0], s[1], self.batch_size, self.num_steps))

        return out


class DatasetIterator:
    """
    Stores a reference to the dataset and produces batches of data.
    """

    def __init__(self, dataset, seq_list):
        """
        Stores a reference to the dataset and produces batches of data.

        Parameters
        ----------
        dataset : Dataset
            an instance of Dataset class
        seq_list : list
            the list of sequences in the dataset to read (different for train_set and test_set)
        """
        self.dataset = dataset
        self.seq_list = seq_list
        self.new_sequence_callbacks = []  # event: list of callables
        self.reset()

    def reset(self):
        self.epochs = 0
        self.iseq = 0
        self.buffer = collections.deque()
        self.ibuf = 0
        self.fire_new_seq = True
        self._fill_buffer()

    def _new_seq_to_buffer(self):
        iseq = self.seq_list[self.iseq]
        new_buffer = self.dataset.format(iseq)  # a list of [[x_0, y_0], [x_1, y_1]] for each sequence
        self.buffer.extend(new_buffer)
        self.buffer_len = len(self.buffer)

        # self.epochs += 1 / len(self.seq_list)
        self.iseq = (self.iseq + 1) % len(self.seq_list)

    def _fill_buffer(self):
        while(len(self.buffer) == 0):
            self._new_seq_to_buffer()

    def produce(self):
        """
        Produce the next batch of (inputs, target)

        Return
        ------
        inputs : numpy ndarray [batch_size x num_steps x input_size]
        targets : numpy ndarray [1 x num_steps x target_size]
        """
        if(self.fire_new_seq):
            [fn() for fn in self.new_sequence_callbacks]
            self.fire_new_seq = False

        assert(len(self.buffer) > 0)
        assert(len(self.buffer[0]) > self.ibuf)
        assert(len(self.buffer[0][self.ibuf]) == 2)

        # print(self.ibuf, '/', len(self.buffer[0]))
        out = self.buffer[0][self.ibuf]
        assert(len(out) == 2)  # x, y
        self.ibuf += 1
        self.epochs += 1 / (len(self.seq_list) * self.buffer_len * len(self.buffer[0]))

        if(len(self.buffer[0]) == self.ibuf):
            self.buffer.popleft()
            self.ibuf = 0
            self.fire_new_seq = True

        if(len(self.buffer) == 0):
            self._fill_buffer()

        return out