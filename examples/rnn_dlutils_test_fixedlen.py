import dataset
import dlutils
import itertools
import numpy as np
import shutil
import os
import pickle
import json

d = dataset.Dataset('./tabs_dict_std_ton2.dat')

# pre-process dataset
x = []
for iseq in range(d.nseq):
    seq = d.y[d.sind[iseq]:d.sind[iseq] + d.slen[iseq]]
    x.append(np.expand_dims(seq, axis=-1))

N = len(x)
models_root = "./test_architecture_ho30"
assert(os.path.isdir(models_root))

# list model dirs
models_dirs = []
for pathname in os.listdir(models_root):
    if(os.path.isdir(pathname)):
        models_dirs.append(pathname)

# load test indices
held_out_pickle_fn = os.path.join(models_root, 'held_out_indices.pickle')
test_ind = pickle.load(open(held_out_pickle_fn, 'rb'))

x_test = [x[i] for i in test_ind]
data_inputs, data_targets = dlutils.dataset.rnn.LM_input_and_targets_from_inputs(x_test, max_len=200)
rnn_dataset = dlutils.dataset.rnn.Dataset_seq2seq(data_inputs, data_targets)

def load_model(models_root, model_dir, vocabulary_size):
    input_stage = dlutils.models.tf_rnn.RNNLM_TF_InputStage_OneHot(vocabulary_size)
    output_stage = dlutils.models.tf_rnn.RNNLM_TF_OutputStage_Classification(vocabulary_size)
    feedback_stage = dlutils.models.tf_rnn.RNNLM_TF_FeedbackStage_Sampler()

    model_meta = json.load(os.path.join(models_root, model_dir, model_dir + '.json'))
    rnn_layers = model_meta["conf"]["rnn_layers"]
    dense_layers = model_meta["conf"]["dense_layers"]

    model = dlutils.models.tf_rnn.RNNLM_TF(input_stage, output_stage, feedback_stage, 
                                           rnn_layers=rnn_layers, 
                                           dense_layers=dense_layers)

    model_ckpt_fn = os.path.join(models_root, model_dir, model_dir + '_0_best.ckpt')
    model.load(model_ckpt_fn)
    return model, rnn_layers


def run_experiment(dataset, list_test_seq, models_root, model_dir, batch_size = 256):
    vocabulary_size = dataset.alphabet_size
    model, rnn_layers = load_model(models_root, model_dir, vocabulary_size)
    di_test = _dataset_rnn.seq2seq_iterator_factory(dataset, seq_list=list_test_seq,
                                                    num_buckets=None, seq_len=4,
                                                    batch_size=batch_size)
    di_test.new_sequence_callbacks.append(model.new_sequence)
    test_loss = epoch_loop(di_test, model.test)
    return test_loss, rnn_layers


out = []
for model_dir in models_dirs:
    seq_ind = np.arange(len(x_test), dtype=np.int)
    test_loss, rnn_layers = run_experiment(rnn_dataset, seq_ind, models_root, model_dir)
    out.append({"test_loss": test_loss, "rnn_layers":rnn_layers})
