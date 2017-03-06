import numpy as np
import math
from collections import deque
from .utils import frame_ndarray


def format_sequence(in_seq, target_seq, batch_size, num_steps, overlap=.5):
    """
    DO NOT USE. DEPRECATED
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


def LM_input_and_targets_from_inputs(x, max_len=None):
    """
    generate input and target sequences as shifted inputs.
    In order to avoid wasting memory, the outputs are sliced references to the input x.
    This means that they will change if you change x.

    Parameters
    ----------
    x: list of ndarray
        each element has dimension [T, ...] where T is the length of the sequence
    max_len: int
        ignore sequences longer than max_len
    """

    x_out = []
    y_out = []
    ignored_sequences = 0
    ignored_max_len = 0
    for i in range(len(x)):
        if(len(x[i]) < 2):
            ignored_sequences += 1
            continue
        if((max_len is not None) and len(x[i]) > max_len):
            ignored_max_len += 1
            continue
        x_out.append(x[i][0:-1])
        y_out.append(x[i][1:])
    if(ignored_sequences > 0):
        print("Warning: %d sequences ignored because length < 2" % (ignored_sequences,))
    if(ignored_max_len > 0):
        print("Warning: %d sequences ignored because length > %d" % (ignored_max_len, max_len))
    return x_out, y_out


class Dataset_seq2seq():
    def __init__(self, x, y=None, name=''):
        """
        Dataset class for variable length sequence to sequence model

        If you have only a list of sequences, use LM_targets_from_inputs
        to generate inpupts and targets using LM_input_and_targets_from_inputs

        Parameters
        ----------
        x: list of ndarray
            each element has dimension [T, ...] where T is the length of the sequence
        y: list of ndarray
            as x
        """
        if(y is None):
            x, y = LM_input_and_targets_from_inputs(x)

        # check number of sequences
        if(len(x) != len(y)):
            raise ValueError("input and targets should have same number of sequences")
        self.nseq = len(x)

        # check sequence lengths
        self.slen = np.zeros(self.nseq, dtype=np.int)
        for i in range(self.nseq):
            if(len(x[i]) != len(y[i])):
                raise ValueError("%d-th input and target sequences have different length " % (i))
            self.slen[i] = len(x[i])

        self.x = x
        self.y = y
        self.name = name

        self.conf = {
            "name": self.name,
            "nseq": self.nseq,
            "tot_len": sum(self.slen.tolist()),
            "max_len": max(self.slen.tolist()),
            "min_len": min(self.slen.tolist()),
        }


def seq2seq_iterator_factory(dataset, seq_list, num_buckets=None, seq_len=None, batch_size=None):
    if((num_buckets is None) and (seq_len is None)):
        raise ValueError('either fixed_length or num_buckets must be specified')
    elif((num_buckets is not None) and (seq_len is not None)):
        raise ValueError('only one between fixed_length and num_buckets must be specified')
    elif(num_buckets is not None):
        return Dataset_seq2seq_iterator(dataset, seq_list, num_buckets=num_buckets, batch_size=batch_size)
    elif(seq_len is not None):
        return Dataset_seq2seq_iterator_fixed_length(dataset, seq_list, seq_len=seq_len, batch_size=batch_size)
    return None


class Dataset_seq2seq_iterator():
    def __init__(self, dataset, seq_list, num_buckets=1, batch_size=None):
        """
        Iterate over seq_list, trying to group sequences of similar lengths

        Parameters
        ----------
        num_buckets: int
            group sequences by length in num_buckets groups
            used for reducing the variability of sequence lengths in the same batch
        """
        if(num_buckets > len(seq_list)):
            raise ValueError("num_buckets should be smaller than number of sequences")

        if(batch_size is not None):
            if(batch_size > len(seq_list) // num_buckets):
                raise ValueError("batch_size should be smaller than number of sequences per bucket")

        self.dataset = dataset
        self.seq_list = np.array(seq_list, dtype=np.int)
        self.batch_size = int(batch_size)
        self.num_buckets = int(num_buckets)
        self.create_buckets()

        d = self.dataset
        self.x_dtype = d.x[0].dtype
        self.x_shape = d.x[0][0].shape
        self.y_dtype = d.y[0].dtype
        self.y_shape = d.y[0][0].shape

        self.new_sequence_callbacks = []  # event: list of callables
        self.epochs = 0

    def create_buckets(self):
        """
        Create self.buckets a list of 1-dim int arrays containing
        the indices of sequences (sorted by length)
        """
        d = self.dataset
        iseq = np.argsort(d.slen[self.seq_list])
        split_points = np.linspace(0, len(self.seq_list), self.num_buckets + 1, endpoint=True, dtype=np.int)
        self.buckets = np.split(iseq, split_points[1:-1])  # indices of seq_list
        self.buckets_counter = np.zeros(self.num_buckets, dtype=np.int)
        self.buckets_size = np.array([len(b_) for b_ in self.buckets], dtype=np.int)

    def shuffle(self):
        # shuffle within each bucket
        for i in range(self.num_buckets):
            np.random.shuffle(self.buckets[i])
            self.buckets_counter[i] = 0

    def next_batch(self, batch_size=None):
        if(batch_size is None):
            batch_size = self.batch_size

        if(np.any(self.buckets_size < batch_size)):
            raise ValueError("batch_size should be smaller than bucket_size")

        # find available buckets:
        available_buckets = np.where(self.buckets_counter + batch_size <= self.buckets_size)[0]
        if(len(available_buckets) == 0):
            self.epochs += 1
            self.shuffle()
            available_buckets = np.arange(self.num_buckets, dtype=np.int)  # all are available

        i = np.random.choice(available_buckets)  # choose from available buckets

        # extract sequences from the i-th bucket
        d = self.dataset
        c = self.buckets_counter[i]
        ib = self.buckets[i][c:c + batch_size]
        ii = self.seq_list[ib]
        x_list = [d.x[ii_] for ii_ in ii]
        y_list = [d.y[ii_] for ii_ in ii]

        self.buckets_counter[i] += batch_size

        # pad sequences to the max length
        lengths = np.array([len(x_) for x_ in x_list], dtype=np.int)
        max_len = np.max(lengths)

        # allocate input and target batch
        x = np.zeros((batch_size, max_len,) + self.x_shape, dtype=self.x_dtype)
        y = np.zeros((batch_size, max_len,) + self.y_shape, dtype=self.y_dtype)

        # copy the sequences
        for i in range(batch_size):
            x[i, :lengths[i], ...] = x_list[i]
            y[i, :lengths[i], ...] = y_list[i]

        # every batch starts a new sequence (TODO: modify for fixed-length truncated BPTT)
        [fn() for fn in self.new_sequence_callbacks]

        return x, y, lengths


class Dataset_seq2seq_iterator_fixed_length():
    def __init__(self, dataset, seq_list, seq_len, batch_size=None):
        """
        Iterate over seq_list 

        Parameters
        ----------
        seq_length: int
            split sequences into seq_len long sub sequences
        """

        self.dataset = dataset
        self.seq_list = np.array(seq_list, dtype=np.int)
        self.iseq = 0  # index of self.seq_list
        self.seq_len = seq_len
        self.batch_size = int(batch_size)

        d = self.dataset
        self.x_dtype = d.x[0].dtype
        self.x_shape = d.x[0][0].shape
        self.y_dtype = d.y[0].dtype
        self.y_shape = d.y[0][0].shape
        self.buffer = deque()

        self.new_sequence_callbacks = []  # event: list of callables
        self.epochs = 0

    def shuffle(self):
        np.random.shuffle(self.seq_list)

    def fill_buffer(self):
        if(self.iseq == len(self.seq_list)):
            self.iseq = 0
            self.epochs += 1
            self.shuffle()
        d = self.dataset
        iseq = self.seq_list[self.iseq]
        x_, y_ = d.x[iseq], d.y[iseq]
        x_framed = frame_ndarray(x_, self.seq_len, 1)
        y_framed = frame_ndarray(y_, self.seq_len, 1)
        self.buffer.extend([(x_f, y_f) for x_f, y_f in zip(x_framed, y_framed)])
        self.iseq += 1

    def next_batch(self, batch_size=None):
        if(batch_size is None):
            batch_size = self.batch_size

        # allocate input and target batch
        x = np.zeros((batch_size, self.seq_len,) + self.x_shape, dtype=self.x_dtype)
        y = np.zeros((batch_size, self.seq_len,) + self.y_shape, dtype=self.y_dtype)
        lengths = np.array([self.seq_len] * batch_size, dtype=np.int)

        remaining = batch_size
        count = 0
        while(remaining > 0):
            if(len(self.buffer) == 0):
                self.fill_buffer()

            toread = min(len(self.buffer), remaining)
            for i in range(toread):
                x_, y_ = self.buffer.popleft()
                x[count, :] = x_
                y[count, :] = y_
                remaining -= 1
                count += 1

        return x, y, lengths
