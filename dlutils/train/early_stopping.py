"""
see Lutz Prechelt "Early Stopping - but when?"
for GeneralizationLoss, TrainingProgress
"""

import numpy as np
import warnings
import collections


class EarlyStoppingBase:
    def __init__(self):
        self.model = None
        self.best_model_path = None
        self.setup_flag = False

    def new_epoch(self, train_error, valid_error):
        return False

    def setup(self, model, best_model_path):
        self.model = model
        self.bost_model_path = best_model_path
        self.setup_flag = True

    def reset(self):
        pass


class GeneralizationLoss:
    def __init__(self):
        self.reset()

    def reset(self):
        self.best_valid_error = np.inf

    def __call__(self, valid_error):
        if(valid_error < self.best_valid_error):
            self.best_valid_error = valid_error
        return (valid_error / self.best_valid_error) - 1.0


class TrainingProgress:
    def __init__(self, lookahead=5):
        self.lookahead = lookahead
        self.reset()

    def reset(self):
        self.train_error = collections.deque([np.inf] * self.lookahead, maxlen=self.lookahead)

    def __call__(self, train_error):
        self.train_error.append(train_error)
        return (np.mean(self.train_error) / np.min(self.train_error)) - 1.0


class Lookahead:
    def __init__(self, lookahead=5):
        self.lookahead = lookahead
        self.reset()

    def reset(self):
        self.flags = collections.deque([False] * self.lookahead, maxlen=self.lookahead)

    def __call__(self, flag):
        self.flags.append(flag)
        return all(self.flags)


class EarlyStopping(EarlyStoppingBase):
    def __init__(self, condition, lookahead=10):
        """
        Stop as soon as condition(train_error, valid_error) is true for lookahead epochs

        Parameters
        ----------
        condition: callable
            implement condition(train_error, valid_error)
            look below for some examples: subclasses of ESCondition
        """
        super().__init__()

        self.condition = condition
        self.lookahead = Lookahead(lookahead)
        self.reset()        

    def reset(self):
        self.lookahead.reset()
        self.condition.reset()
        self.best_valid_error = np.inf

    def new_epoch(self, train_error, valid_error):
        if(not self.setup_flag):
            warnings.warn("EarlyStoppingValid Warning: no model or path for saving. \
                You should call setup before start training")

        if(valid_error < self.best_valid_error):
            self.best_valid_error = valid_error
            self.model.save(self.bost_model_path)

        return self.lookahead(self.condition(train_error, valid_error))


class ESCondition():
    def reset(self):
        pass

    def __call__(self, train_error, valid_error):
        return False


class ESCondition_0(ESCondition):
    def __init__(self, alpha=0.01):
        """
        Stop as soon as GL > alpha
        This is good to check for overfitting behavior

        Parameters
        ----------
        alpha: float
            small tolerance
        """
        self.gl = GeneralizationLoss()
        self.alpha = alpha

    def reset(self):
        self.gl.reset()

    def __call__(self, train_error, valid_error):
        gl = self.gl(valid_error)
        return gl > self.alpha


class ESCondition_1(ESCondition):
    def __init__(self, alpha=0.01, training_progress_lookahead=10):
        """
        Stop as soon as GL / TP > alpha
        where GL: generalization loss and TP: training progress

        Parameters
        ----------
        alpha: float
            small tolerance
        training_progress_lookahead: int
            lookahead for training_progress measure
        """
        self.gl = GeneralizationLoss()
        self.tp = TrainingProgress(training_progress_lookahead)
        self.alpha = alpha

    def reset(self):
        self.gl.reset()
        self.tp.reset()

    def __call__(self, train_error, valid_error):
        gl = self.gl(valid_error)
        tp = self.tp(train_error)
        return (gl / tp) > self.alpha


class ESCondition_2(ESCondition):
    def __init__(self, alpha=0.05, lookahead=5):
        """
        Stop as soon as std(valid_error) / mean(valid_error) < alpha
        Check for constant regions (convergence)

        Parameters
        ----------
        alpha: float
            small tolerance
        lookahead: int
            moving window length
        """
        self.alpha = alpha
        self.lookahead = lookahead
        self.reset()

    def reset(self):
        self.valid_error = collections.deque([np.inf] * self.lookahead, maxlen=self.lookahead)

    def __call__(self, train_error, valid_error):
        self.valid_error.append(valid_error)
        return (np.std(self.valid_error) / np.mean(self.valid_error)) < self.alpha
