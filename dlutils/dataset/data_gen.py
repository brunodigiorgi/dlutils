import numpy as np
import itertools

# TODO: generate real valued sequences: totally random and with some dependencies (gaussians)


def gen_data_seq2seq(size, in_size, out_size, deps_list=[]):
    """
    Create a sequence of given size with given conditional probability
    Default is uniform probability

    Parameters
    ----------
    size : int
        length of the dataset
    in_size : int
        number of input symbols
    out_size : int
        number of output symbols
    deps_list : list of dict
        {'delay': int, 'in_symbol': int, 'out_symbol': int, 'add_prob': float [0, 1]}
        limitation: 'delay' must be different for each element

    Results
    -------
    X : np.array
        input data
    Y : np.array
        target_data
    """

    # check
    delays = [d['delay'] for d in deps_list]
    assert(len(delays) == len(set(delays)))

    X = np.array(np.random.choice(in_size, size=(size,)), dtype=np.int32)
    Y = np.zeros([size], dtype=np.int32)
    for i in range(size):
        # initialize with uniform distribution
        y_prob = np.ones([out_size]) / out_size
        for r in deps_list:
            if((i >= r['delay']) and (X[i - r['delay']] == r['in_symbol'])):
                y_prob[r['out_symbol']] += r['add_prob']
                y_prob[np.arange(out_size) != r['out_symbol']
                       ] -= r['add_prob'] / (out_size - 1)
        Y[i] = np.random.choice(out_size, p=y_prob)
    return X, Y


def expected_cross_entropy(in_size, out_size, deps_list):
    """
    Compute the expected cross-entropy loss, given a number of dependencies and input/output alphabet size

    Parameters
    ----------
    see gen_data_seq2seq

    Returns
    -------
    out : list
        out[i] is the expected cross-entropy when 0,...,i-th dependencies have been learned

    """
    num_deps = len(deps_list)

    # create the grid of probability with respect to all input dependencies
    # configuration
    dims = np.concatenate([np.array([in_size] * num_deps,
                                    dtype=np.int32), np.array([out_size], dtype=np.int)])
    M = np.ones(dims) / out_size
    it = [list(range(in_size))] * num_deps
    for in_conf in itertools.product(*it):
        for i, dep in enumerate(deps_list):
            if(in_conf[i] == dep['in_symbol']):
                o = dep['out_symbol']
                M[in_conf][o] += dep['add_prob']
                M[in_conf][np.arange(out_size) !=
                           o] -= dep['add_prob'] / (out_size - 1)

    # compute cross entropy for every number of dependencies learned
    out = []
    for learned_deps in range(num_deps + 1):
        it = [list(range(in_size))] * learned_deps
        p = []
        for in_conf in itertools.product(*it):
            p.append(np.mean(np.reshape(M[in_conf], [-1, out_size]), axis=0))
        # cross-entropy
        p = np.array(p)
        cross_ent = -np.mean(np.sum(p * np.log(p + 1e-10), axis=-1))
        out.append(cross_ent)
    return out
