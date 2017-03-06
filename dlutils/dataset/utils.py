import numpy as np
import collections


def frame_ndarray(a, frame_size, hop_size):
    """
    Create slices of the input array along the first dimension, with given frame_size and hop_size
    The other dimensions are preserved

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
        last entries might be neglected:
        nframes = 1 + (a.shape[0] - frame_size) // hop_size
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
