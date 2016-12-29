from sklearn import cross_validation
import time
from random import shuffle
from .dataset import DatasetIterator
from time import localtime, strftime
import os
import math


class RNNTest:
    def __init__(self, dataset, model, batch_size, logger,
                 nfolds=5, max_fold=None, proportion=1, update_interval=120,
                 log_epochs=1, max_epochs=1000, save_every=100, model_path='models/'):
        self.dataset = dataset
        self.model = model
        self.batch_size = batch_size
        self.logger = logger
        self.nfolds = nfolds
        assert(self.nfolds > 0)
        self.max_fold = max_fold
        if(self.max_fold is None):
            self.max_fold = self.nfolds
        assert((self.max_fold > 0) and (self.max_fold <= self.nfolds))
        self.proportion = proportion
        self.update_interval = update_interval
        self.log_epochs = log_epochs
        self.max_epochs = max_epochs
        self.model_path = model_path
        if(not os.path.exists(self.model_path)):
            os.mkdir(self.model_path)

        self.conf = {
            "batch_size": self.batch_size,
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

    def test(self):
        self.logger.new_model(self.conf, self.model_id)
        log_time = time.time()
        for i_fold, (train_set, test_set) in enumerate(self.k_fold):
            if(i_fold >= self.max_fold):
                continue
            self.logger.new_fold()
            self.model.initialize()

            list_train_seq = [self.list_seq[i] for i in train_set]
            list_test_seq = [self.list_seq[i] for i in test_set]
            di_train = DatasetIterator(self.dataset, list_train_seq)
            di_test = DatasetIterator(self.dataset, list_test_seq)
            di_train.new_sequence_callbacks.append(self.model.new_sequence)  # callback to reset the rnn state

            log_epoch = 0
            save_epoch = 0
            for iepoch in range(self.conf['max_epochs']):
                self.model.set_epoch(iepoch)

                train_ev_, count = 0, 0
                while(di_train.epochs < iepoch + 1):
                    inputs_, targets_ = di_train.produce()
                    train_ev__ = self.model.train(inputs_, targets_)
                    train_ev_ += train_ev__
                    count += 1
                    # print("epoch:", iepoch, count, "train_loss:", train_ev__, di_train.epochs)

                    if(time.time() > log_time + self.update_interval):
                        log_time = time.time()
                        self.logger.set_current_epoch(di_train.epochs)

                train_ev = train_ev_ / count
                if(di_train.epochs >= log_epoch + self.log_epochs):
                    log_epoch = math.floor(di_train.epochs)
                    self.logger.append_train_ev(train_ev, log_epoch - 1)

                    # test
                    if(self.nfolds > 1):
                        test_ev_, count = 0, 0
                        while(di_test.epochs < iepoch + 1):
                            inputs_, targets_ = di_test.produce()
                            test_ev_ += self.model.test(inputs_, targets_)
                            count += 1
                        test_ev = test_ev_ / count
                        self.logger.append_test_ev(test_ev, log_epoch - 1)

                if(iepoch >= save_epoch + self.conf["save_every"]):
                    model_fn = self.model_id + '_' + str(i_fold) + "_" + str(iepoch) + ".ckpt"
                    model_fn = os.path.join(self.model_path, model_fn)
                    self.model.save(model_fn)
                    save_epoch += self.conf["save_every"]
