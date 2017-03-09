import dataset
import dlutils
import itertools
import numpy as np
import shutil
import os
import pickle

d = dataset.Dataset('./tabs_dict_std_ton2.dat')

# pre-process dataset
x = []
for iseq in range(d.nseq):
    seq = d.y[d.sind[iseq]:d.sind[iseq] + d.slen[iseq]]
    x.append(np.expand_dims(seq, axis=-1))

N = len(x)
hold_out = .3
model_path = "./test_architecture_ho%.0f" % (hold_out * 100,)
held_out_pickle_fn = os.path.join(model_path, 'held_out_indices.pickle')

ind = np.arange(N, dtype=np.int)
if(not os.path.isfile(held_out_pickle_fn)):
    np.random.shuffle(ind)
    split_ind = int(N * hold_out)
    train_ind = ind[split_ind:]
    test_ind = ind[:split_ind]
    pickle.dump(test_ind, open(held_out_pickle_fn, 'wb'))
else:
    test_ind = pickle.load(open(held_out_pickle_fn, 'rb'))
    train_ind = np.setdiff1d(ind, test_ind)

x_train = [x[i] for i in train_ind]
data_inputs, data_targets = dlutils.dataset.rnn.LM_input_and_targets_from_inputs(x_train, max_len=200)
rnn_dataset = dlutils.dataset.rnn.Dataset_seq2seq(data_inputs, data_targets)

if(not os.path.isdir(model_path)):
    os.mkdir(model_path)

# constant parameters
const_conf = {
    "batch_size": 256,
    "vocabulary_size": d.alphabet_size,
    "dense_layers": [{"num_units": d.alphabet_size}],
    "keep_prob": 1,
    "nfolds": 5,
    "max_epochs": 3000,
    "num_buckets": 12,  # was 20
    "fixed_length": None,  # was 4
    "proportion": 1,
    "es_condition_alpha": 0.01,
    "rnn_cell_type": "LSTM",
}

# hyperparameters
hyper_conf = {
    "rnn_layers": [
        [32], [32, 32], [32, 32, 32],
        [64], [64, 64], [64, 64, 64],
        [128], [128, 128], [128, 128, 128],
    ],
    "learning_rate": [0.01],
    "optimizer": ["RMSProp"],
}

# generate all the configurations, mixing constant- and hyper-parameters
keys = list(hyper_conf.keys())
values = list(hyper_conf.values())
conf = []
for v in itertools.product(*values):
    conf_ = const_conf.copy()
    for k, v in zip(keys, v):
        conf_[k] = v
    conf.append(conf_)


def clear_folder(folder):
    print("removing old content of folder:")
    if(not os.path.isdir(folder)):
        return
    for f in os.listdir(folder):
        if(f == ".DS_Store"):
            continue
        file_path = os.path.join(folder, f)
        print("  ", file_path)
        if(os.path.isdir(file_path)):
            shutil.rmtree(file_path)
        elif(os.path.isfile(file_path)):
            os.remove(file_path)
        else:
            raise ValueError("clear folder error for file %s" % (file_path,))


def run_experiment(conf_):
    batch_size = conf_["batch_size"]
    vocabulary_size = conf_["vocabulary_size"]
    dense_layers = conf_["dense_layers"]
    keep_prob = conf_["keep_prob"]
    nfolds = conf_["nfolds"]
    max_epochs = conf_["max_epochs"]
    num_buckets = conf_["num_buckets"]
    fixed_length = conf_["fixed_length"]
    proportion = conf_["proportion"]
    es_condition_alpha = conf_["es_condition_alpha"]
    rnn_layers = [{"num_units": nu, "cell_type": conf_["rnn_cell_type"]}
                  for nu in conf_["rnn_layers"]]
    optimizer = conf_["optimizer"]
    learning_rate = conf_["learning_rate"]

    input_stage = dlutils.models.tf_rnn.RNNLM_TF_InputStage_OneHot(
        vocabulary_size)
    output_stage = dlutils.models.tf_rnn.RNNLM_TF_OutputStage_Classification(
        vocabulary_size)
    feedback_stage = dlutils.models.tf_rnn.RNNLM_TF_FeedbackStage_Sampler()
    model = dlutils.models.tf_rnn.RNNLM_TF(input_stage, output_stage, feedback_stage, rnn_layers, dense_layers,
                                           optimizer=optimizer, learning_rate=learning_rate,
                                           keep_prob=keep_prob, max_to_keep=None)

    logger = dlutils.train.HtmlLogger()
    es_condition_0 = dlutils.train.early_stopping.ESCondition_0(alpha=es_condition_alpha)
    es_condition_2 = dlutils.train.early_stopping.ESCondition_2(alpha=es_condition_alpha, lookahead=5)
    es_condition = dlutils.train.early_stopping.ESCondition_any([es_condition_0, es_condition_2])
    early_stopping = dlutils.train.early_stopping.EarlyStopping(es_condition, lookahead=5)
    trainer = dlutils.train.rnn.Trainer(rnn_dataset, model,
                                        batch_size=batch_size, num_buckets=num_buckets, fixed_length=fixed_length,
                                        logger=logger,
                                        nfolds=nfolds, max_fold=nfolds, proportion=proportion,
                                        max_epochs=max_epochs, early_stopping=early_stopping, model_path=model_path, save_every=20)
    trainer.train()


clear_folder(model_path)
for conf_ in conf:
    run_experiment(conf_)
