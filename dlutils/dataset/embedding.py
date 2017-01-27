from .utils import DatasetTranslator


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