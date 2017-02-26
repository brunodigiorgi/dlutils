from sklearn.model_selection import KFold
import numpy as np
from time import localtime, strftime
import os
import math
import importlib
from ..dataset import rnn as _dataset_rnn
from . import logger as _logger
from . import early_stopping as _early_stopping

# reload modules from this package (to see changes while in interactive development sessions)
_dataset_rnn = importlib.reload(_dataset_rnn)
_logger = importlib.reload(_logger)
_early_stopping = importlib.reload(_early_stopping)


def epoch_loop(dataset_iterator, model_fn, log_fn=None):
    loss, count = 0, 0
    iepoch = math.floor(dataset_iterator.epochs)
    while(dataset_iterator.epochs < iepoch + 1):
        inputs_, targets_, seqlen_ = dataset_iterator.next_batch()
        loss_ = model_fn(inputs_, targets_, seqlen_)
        loss += loss_
        count += 1
        if(log_fn is not None):
            log_fn(loss_, dataset_iterator.epochs)
    return loss / count


class Trainer:
    def __init__(self, dataset, model, batch_size, num_buckets=1, logger=None,
                 nfolds=5, max_fold=None, proportion=1,
                 max_epochs=1000, early_stopping=None, save_every=100, model_path='models/'):
        self.dataset = dataset
        self.model = model
        self.batch_size = int(batch_size)
        self.num_buckets = int(num_buckets)

        self.logger = logger
        if(self.logger is None):
            self.logger = _logger.Logger()

        self.nfolds = nfolds
        assert(self.nfolds > 0)
        self.max_fold = max_fold
        if(self.max_fold is None):
            self.max_fold = self.nfolds

        assert((self.max_fold > 0) and (self.max_fold <= self.nfolds))

        self.early_stopping = early_stopping
        if(self.early_stopping is None):
            self.early_stopping = _early_stopping.EarlyStoppingBase()  # dummy early stopping

        self.save_every = save_every
        self.proportion = proportion
        self.max_epochs = max_epochs
        self.model_path = model_path
        if(not os.path.exists(self.model_path)):
            os.mkdir(self.model_path)

        self.conf = {
            "save_every": self.save_every,
            "max_epochs": self.max_epochs,
            "proportion": self.proportion,
            "nfolds": self.nfolds,
            "batch_size": self.batch_size,
        }
        self.conf["model"] = self.model.conf
        self.conf["dataset"] = self.dataset.conf

        self.model_id = strftime("%Y%m%d%H%M%S", localtime())
        self.model_dir = os.path.join(self.model_path, self.model_id)
        if(not os.path.exists(self.model_dir)):
            os.mkdir(self.model_dir)

        self.list_seq = np.arange(int(dataset.nseq * self.proportion), dtype=np.int).reshape(-1, 1)
        if(self.nfolds > 1):
            kf = KFold(n_splits=nfolds, shuffle=True)
            self.k_fold = list(kf.split(self.list_seq))
        else:
            self.k_fold = [(self.list_seq[:, 0], self.list_seq[:, 0])]

    def train(self, contd=False):
        self.logger.new_model(self.conf, self.model_dir, self.model_id)

        # k-fold cross-validation
        for i_fold, (train_set, test_set) in enumerate(self.k_fold):
            print("  fold:", i_fold)
            if(i_fold >= self.max_fold):
                continue
            self.logger.new_fold()

            if(contd):
                if(not self.model.initialized):
                    raise ValueError("using contd=True with a non initialized model")
            else:
                self.model.initialize()

            list_train_seq = self.list_seq[train_set, 0]
            list_test_seq = self.list_seq[test_set, 0]
            di_train = _dataset_rnn.Dataset_seq2seq_iterator(self.dataset, seq_list=list_train_seq, batch_size=self.batch_size, num_buckets=self.num_buckets)
            di_test = _dataset_rnn.Dataset_seq2seq_iterator(self.dataset, seq_list=list_test_seq, batch_size=self.batch_size, num_buckets=self.num_buckets)

            # callback to reset the rnn state
            di_train.new_sequence_callbacks.append(self.model.new_sequence)
            di_test.new_sequence_callbacks.append(self.model.new_sequence)

            # early stopping
            best_model_path = self.model_id + '_' + str(i_fold) + "_best.ckpt"
            best_model_path = os.path.join(self.model_dir, best_model_path)
            self.early_stopping.setup(self.model, best_model_path)
            self.early_stopping.reset()

            log_epoch = 0
            save_epoch = 0
            for iepoch in range(self.conf['max_epochs']):
                self.logger.new_epoch(iepoch)

                # train epoch
                train_loss = epoch_loop(di_train, self.model.train)
                log_epoch = math.floor(di_train.epochs)
                self.logger.train_epoch(train_loss, log_epoch)

                # test epoch
                if(self.nfolds > 1):
                    test_loss = epoch_loop(di_test, self.model.test)
                    self.logger.test_epoch(test_loss, log_epoch)
                else:
                    test_loss = train_loss

                # save
                if(iepoch >= save_epoch + self.conf["save_every"]):
                    model_fn = self.model_id + '_' + str(i_fold) + "_" + str(iepoch) + ".ckpt"
                    model_fn = os.path.join(self.model_dir, model_fn)
                    self.model.save(model_fn)
                    save_epoch += self.conf["save_every"]

                # early stopping
                stop = self.early_stopping.new_epoch(train_loss, test_loss)
                if(stop):
                    break
