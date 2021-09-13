import torch
import torch.nn.functional as F
from torch import nn

from my_common.my_helper import is_positive_int
from util.constants import Activation, TaskType, LossType, Regularization


def last_dims_tuple(xx, k=1):
    assert torch.is_tensor(xx)
    return tuple(range(k, xx.ndim))


def get_act_func(act):
    """
    Return the corresponding torch activation function given its name
    """
    assert isinstance(act, Activation)
    if act == Activation.ReLU:
        return F.relu  # Used in RP-Paper and original Set Transformer
    elif act == Activation.Tanh:
        return torch.tanh  # torch.functional.tanh is deprecated
    elif act == Activation.Sigmoid:
        return torch.sigmoid
    else:
        raise NotImplementedError(f"Haven't yet implemented models with {act.name} activation.")


def get_loss_func(task_type, loss_type=None):
    """
    Return the loss function. If loss type is given, return the corresponding loss function;
    otherwise return the default loss function associated with the given task type.
    """
    assert isinstance(task_type, TaskType)

    if loss_type is not None:
        assert isinstance(loss_type, LossType)
        if loss_type == LossType.mse:
            assert task_type == TaskType.regression
            return nn.MSELoss(reduction='none')
        elif loss_type == LossType.mae:
            assert task_type == TaskType.regression
            return nn.L1Loss(reduction='none')
        else:
            raise NotImplementedError(f"Haven't yet implemented models with {loss_type.name} loss.")
    else:
        if task_type == TaskType.binary_classification:
            return nn.BCEWithLogitsLoss(reduction='none')
        elif task_type in (TaskType.multi_classification, TaskType.node_classification):
            return nn.CrossEntropyLoss(reduction='none')
        elif task_type == TaskType.regression:
            return nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError(f"Haven't yet implemented models for {task_type.name} task.")


class Two_Phase_Regularization_Scheduler:

    def __init__(self, cfg):
        assert isinstance(cfg.pull_down, bool)
        assert is_positive_int(cfg.milestone)
        assert isinstance(cfg.regularization, Regularization)

        self.pull_down = cfg.pull_down
        self.milestone = cfg.milestone
        self.regularization = cfg.regularization

    def step(self, model, epoch):
        if self.pull_down:
            if epoch >= self.milestone and model.regularizer.regularization != Regularization.none:
                model.regularizer.regularization = Regularization.none
        elif epoch <= self.milestone:  # when epoch > self.milestone, model.regularization is turned on by default
            if epoch == self.milestone:
                model.regularizer.regularization = self.regularization
            else:
                model.regularizer.regularization = Regularization.none
