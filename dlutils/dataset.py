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


def format_sequence(seq, batch_size, num_steps, overlap=.5):
    # print('format_sequence', batch_size, num_steps)
    data_len = len(seq)
    batch_len = data_len // batch_size
    hop_size = batch_len

    if(overlap is not None):
        last_batch = batch_size - 1
        batch_len = - data_len / (overlap * last_batch - last_batch - 1)
        hop_size = batch_len * (1 - overlap)
        hop_size = math.floor(hop_size)
        batch_len = data_len - hop_size * last_batch

    # make sure that hop_size is not a multiple of num_steps (for variety in batches)
    # Effectiveness not yet proved
    if(hop_size % num_steps == 0):
        hop_size -= 1
        batch_len = data_len - hop_size * last_batch

    seq_reshaped = frame_ndarray(seq[0:(hop_size * batch_size + batch_len)], batch_len, hop_size)
    nbatches = (batch_len - 1) // num_steps
    out = []
    for i in range(nbatches):
        x_ = seq_reshaped[:, i * num_steps:(i + 1) * num_steps, ...]
        y_ = seq_reshaped[:, i * num_steps + 1:(i + 1) * num_steps + 1, ...]
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


class DatasetLanguageModel(Dataset):
    def __init__(self, data, batch_size, num_steps, dataset_transformation=None, transformation_mode='pre', dataset_augmentations=[]):
        """
        Parameters
        ----------
        data : string or list of numpy ndarray
            if string is a npz file containing a list of ndarray [seq_length x frame_size]
        dataset_transformation : callable
            applied to every original or augmented sequence
        transformation_mode : 'pre' | 'post'
            select if the transformation is applied before or after the augmentations
        dataset_augmentations : list of callable
            create a new sequence from a given one
        """
        self.data = data
        if(isinstance(data, str)):
            self.data = pickle.load(open(self.data, 'rb'))

        self.batch_size = batch_size
        self.num_steps = num_steps
        self.dataset_transformation = dataset_transformation
        self.transformation_mode = transformation_mode
        self.dataset_augmentations = dataset_augmentations

        self.nseq = len(self.data)

        if(len(self.data[0].shape) == 1):
            self.input_size = 1
            for i, d in enumerate(self.data):
                self.data[i] = self.data[i].reshape(-1, 1)
        else:
            self.input_size = self.data[0].shape[1]

        self.target_size = self.input_size  # for language model the target shares the input domain
        self.nseq = len(self.data)
        self.slen = [len(s) for s in self.data]

        self.conf = {
            "nseq": self.nseq,
            "tot_len": sum(self.slen),
            "max_len": max(self.slen),
            "min_len": min(self.slen),
            "batch_size": self.batch_size,
            "num_steps": self.num_steps
        }

    def format(self, iseq):
        seq = self.data[iseq]

        if(self.dataset_transformation is not None and self.transformation_mode == 'pre'):
            seq = self.dataset_transformation(seq)

        seqs = [seq]
        for a in self.dataset_augmentations:
            augmented_seqs = a(seq)
            seqs.extend(augmented_seqs)

        if(self.dataset_transformation is not None and self.transformation_mode == 'post'):
            nseqs = len(seqs)
            for i in range(nseqs):
                seqs[i] = self.dataset_transformation(seqs[i])

        out = []
        for s in seqs:
            out.append(format_sequence(s, self.batch_size, self.num_steps))

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

        self.alphabet = sorted(list(self.count.keys()))
        self._to_int = {symbol: i for i, symbol in enumerate(self.alphabet)}
        self._to_symbol = {i: symbol for i, symbol in enumerate(self.alphabet)}

        self.unknown_int = None
        if(vocabulary_size is not None):
            self.unknown_int = self._to_int['UNK']

    def to_int(self, data):
        """
        Parameters
        ----------
        data: list [of lists]+ of symbols, or just a symbol
        """
        try:  # assume iterable
            assert(not isinstance(data, str))
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
        self.epochs = - 1 / len(self.seq_list)
        self.iseq = 0
        self.buffer = collections.deque()
        self.ibatch = 0
        self.fire_new_seq = True
        self._fill_buffer()

    def _new_seq_to_buffer(self):
        iseq = self.seq_list[self.iseq]
        new_buffer = self.dataset.format(iseq)  # a list of [[x_0, y_0], [x_1, y_1]] for each sequence
        self.buffer.extend(new_buffer)

        self.epochs += 1 / len(self.seq_list)
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

        out = self.buffer[0][self.ibatch]
        self.ibatch += 1

        if(self.ibatch == len(self.buffer[0])):
            self.buffer.popleft()
            self.ibatch = 0
            self.fire_new_seq = True

        if(len(self.buffer) == 0):
            self._fill_buffer()

        return out

        # if(self.batch_size is None):
        #     out = [b[:] for b in self.buffer]
        #     self.buffer = None
        # else:
        #     out = [b[:self.batch_size] for b in self.buffer]
        #     self.buffer = [b[self.batch_size:] for b in self.buffer]
        #     self._update_len()
        # self._fill_buffer()
        # return out
