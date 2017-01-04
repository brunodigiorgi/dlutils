from sklearn import cross_validation
import time
from random import shuffle
from .dataset import DatasetIterator
from .logger import Logger
from time import localtime, strftime
import os
import math


def epoch_loop(dataset_iterator, model_fn, log_fn):
    loss, count = 0, 0
    iepoch = math.floor(dataset_iterator.epochs)
    while(dataset_iterator.epochs <= iepoch + 1):
        inputs_, targets_ = dataset_iterator.produce()
        loss_ = model_fn(inputs_, targets_)
        loss += loss_
        count += 1
        log_fn(loss_, dataset_iterator.epochs)
    return loss / count


class RNNTest:
    def __init__(self, dataset, model, logger=None,
                 nfolds=5, max_fold=None, proportion=1,
                 max_epochs=1000, save_every=100, model_path='models/'):
        self.dataset = dataset
        self.model = model
        self.logger = logger
        if(self.logger is None):
            self.logger = Logger()
        self.nfolds = nfolds
        assert(self.nfolds > 0)
        self.max_fold = max_fold
        if(self.max_fold is None):
            self.max_fold = self.nfolds
        assert((self.max_fold > 0) and (self.max_fold <= self.nfolds))
        self.proportion = proportion
        self.max_epochs = max_epochs
        self.model_path = model_path
        if(not os.path.exists(self.model_path)):
            os.mkdir(self.model_path)

        self.conf = {
            "save_every": save_every,
            "max_epochs": self.max_epochs,
            "proportion": self.proportion,
            "nfolds": self.nfolds,
            "model": {},
            "dataset": {}
        }
        self.conf["model"].update(self.model.conf)
        self.conf["dataset"].update(self.dataset.conf)

        self.model_id = strftime("%Y%m%d%H%M%S", localtime())

        self.list_seq = list(range(int(dataset.nseq * self.proportion)))
        shuffle(self.list_seq)
        if(self.nfolds > 1):
            self.k_fold = cross_validation.KFold(n=len(self.list_seq), n_folds=nfolds)
        else:
            self.k_fold = [(self.list_seq, self.list_seq)]

    def test(self, contd=False):
        self.logger.new_model(self.conf, self.model_id)

        for i_fold, (train_set, test_set) in enumerate(self.k_fold):
            if(i_fold >= self.max_fold):
                continue
            self.logger.new_fold()
            if(not contd):
                self.model.initialize()

            list_train_seq = [self.list_seq[i] for i in train_set]
            list_test_seq = [self.list_seq[i] for i in test_set]
            di_train = DatasetIterator(self.dataset, list_train_seq)
            di_test = DatasetIterator(self.dataset, list_test_seq)
            di_train.new_sequence_callbacks.append(self.model.new_sequence)  # callback to reset the rnn state

            log_epoch = 0
            save_epoch = 0
            for iepoch in range(self.conf['max_epochs']):
                self.logger.new_epoch(iepoch)

                # train epoch
                train_loss = epoch_loop(di_train, self.model.train, self.logger.train_step)
                log_epoch = math.floor(di_train.epochs)
                self.logger.train_epoch(train_loss, log_epoch)

                # test epoch
                if(self.nfolds > 1):
                    test_loss = epoch_loop(di_test, self.model.test, self.logger.test_step)
                    self.logger.test_epoch(test_loss, log_epoch - 1)

                # save
                if(iepoch >= save_epoch + self.conf["save_every"]):
                    model_fn = self.model_id + '_' + str(i_fold) + "_" + str(iepoch) + ".ckpt"
                    model_fn = os.path.join(self.model_path, model_fn)
                    self.model.save(model_fn)
                    save_epoch += self.conf["save_every"]
