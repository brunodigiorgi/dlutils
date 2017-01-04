import pickle
import numpy as np
import random
import collections
import math


def frame_ndarray(a, frame_size, hop_size):
    """
    Create slices of the input array along the first dimension, with given frame_size and hop_size

    Parameters
    ----------
    a : ndarray
        input array
    frame_size : int
    hop_size : int

    Return
    ------
    out : ndarray
        out.shape = [nframes, frame_size] + a.shape[1:]
    """
    n = a.shape[0]
    nframes = 1 + (n - frame_size) // hop_size
    other_dim = a.shape[1:]
    if(nframes < 0):
        nframes = 0
    b = np.zeros([nframes, frame_size] + list(other_dim), dtype=a.dtype)

    for i in range(nframes):
        b[i] = a[i * hop_size: i * hop_size + frame_size]
    return b


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


class DatasetAugmentation:
    def __init__(self):
        pass

    def __call__(self, seq):
        """
        Transform an input sequence into other sequences useful for learning data invariances.

        Parameters
        ----------
        seq : numpy ndarray
            input sequence [nframes x frame_size]

        Return
        ------
        seqs : list of numpy ndarray
            a list of all the transformed sequences
        """
        return []


class OneHotEncoder:
    """
    Callable, given a sequence of int returns the one hot representation as a numpy matrix with shape [len(seq), size]
    """
    def __init__(self, size):
        self.size = size
        self.eye = np.eye(self.size)

    def __call__(self, seq):
        assert(seq.shape[1] == 1)
        return self.eye[np.array(seq, dtype=int)[:, 0]]


def one_hot_decode(X):
    """
    Parameters
    ----------
    X : numpy ndarray
        [num_points, alphabet_size]

    Returns
    -------
    out : list
        list of int
    """
    out = []
    for x_ in X:
        out.append(np.argmax(x_))
    return out


class Dataset:
    def format(self, iseq):
        """
        Transform a single sequence corresponding to the given index [iseq]
        into a list of sequences of [num_steps] frames, target_frame for each sequence
        """
        raise NotImplemented()
        pass


class Dataset_seq2seq(Dataset):
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
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.target_data = target_data
        self.name = name
        self.seq_processor = seq_processor

        self.nseq = len(self.data)
        if(self.data[0].ndim == 1):
            self.input_size = 1
            for i, d in enumerate(self.data):
                self.data[i] = self.data[i].reshape(-1, 1)
        else:
            self.input_size = self.data[0].shape[1]

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
            "batch_size": self.batch_size,
            "num_steps": self.num_steps,
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


class DatasetTranslator:
    """
    Translate from given symbols (data is a list of lists of symbols) to int.
    """
    def __init__(self, data, vocabulary_size=None):
        count = collections.Counter()
        for in_list in data:
            count.update(in_list)

        if(vocabulary_size is not None):
            self.count = dict(count.most_common(vocabulary_size - 1))
            self.count['UNK'] = sum(count.values()) - sum(self.count.values())
        else:
            self.count = dict(count)

        x = np.array(list(self.count.values()), dtype=np.float32)
        x = x / np.sum(x)
        self.entropy = - np.sum(x * np.log(x))

        self.alphabet = sorted(list(self.count.keys()))
        self._to_int = {symbol: i for i, symbol in enumerate(self.alphabet)}
        self._to_symbol = {i: symbol for i, symbol in enumerate(self.alphabet)}

        self.unknown_int = None
        if(vocabulary_size is not None):
            self.unknown_int = self._to_int['UNK']

        self.conf = {
            "entropy": self.entropy,
            "nsymbols": len(self.alphabet),
            "alphabet": self.alphabet
        }

    def to_int(self, data):
        """
        Parameters
        ----------
        data: list [of lists]+ of symbols, or just a symbol
        """
        try:  # assume iterable
            assert(not isinstance(data, str))  # do not iterate if string
            out = []
            for d in data:
                out.append(self.to_int(d))
            return out
        except:  # not iterable, assume int
            return self._to_int.get(data, self.unknown_int)

    def to_symbol(self, data):
        """
        Parameters
        ----------
        data: list [of lists]+ of ints, or just a int
        """
        try:  # assume iterable
            out = []
            for d in data:
                out.append(self.to_symbol(d))
            return out
        except:  # not iterable, assume int
            return self._to_symbol[data]

    def print_counts(self):
        import operator
        sorted_counts = sorted(self.count.items(), key=operator.itemgetter(1))
        print(sorted_counts)


def generate_skipgram(seq, num_skips, skip_window):
    """
    generate input and targets by sliding a window on the sequence
    input is the center of the sequence,
    for each input, create [num_skips] targets that are in a neighborhood of the center
    this neighborhood is [skip_window] wide

    Parameters
    ----------
    seq : list of ints
    num_skips : int
        num of targets for each symbol
    skip_window : int
        how far the target can be picked from the input

    taken from tensorflow word2vec_basic.py
    """
    assert num_skips <= 2 * skip_window
    inputs = []
    labels = []
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    fifo = collections.deque(maxlen=span)
    for i in range(span):
        fifo.append(seq[i])

    i = span
    while(i < len(seq)):
        target = skip_window  # target label at the center of the fifo
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            inputs.append(fifo[skip_window])
            labels.append(fifo[target])
        fifo.append(seq[i])
        i += 1
    return inputs, labels


class DatasetEmbedding:
    def __init__(self, data, vocabulary_size=None, mode="skipgram", num_skips=2, skip_window=1):
        """
        Parameters
        ----------
        data : list of lists of symbols
        mode : string "CBOW"|"skipgram"
        """

        self.data = data
        self.vocabulary_size = vocabulary_size
        self.mode = mode
        self.num_skips = num_skips
        self.skip_window = skip_window

        self.translator = DatasetTranslator(data, vocabulary_size)
        self.data_int = self.translator.to_int(data)

        self.nseq = len(self.data)
        self.slen = [len(s) for s in self.data]

        self.conf = {
            "nseq": self.nseq,
            "tot_len": sum(self.slen),
            "max_len": max(self.slen),
            "min_len": min(self.slen),
        }

    def format(self, iseq):
        seq = self.data[iseq]

        if(self.mode == "skipgram"):
            inputs, targets = generate_skipgram(seq, self.num_skips, self.skip_window)
        else:
            raise NotImplemented("Not Yet!")

        inputs = np.array(inputs)  # for embedding lookups
        targets = np.expand_dims(np.array(targets), axis=1)  # for target
        return inputs, targets


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