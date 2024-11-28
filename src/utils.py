import os
from typing import List
import numpy as np
import torch
import torchmetrics
from transformer_lens.hook_points import HookPoint

class Last10Accuracy():
    def __init__(self, nb_classes, device="cpu"):
        self.multiclass_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=nb_classes)

    def to(self, device):
        self.multiclass_acc.to(device)

    def __call__(self, y_pred, y):
        y_pred = y_pred[:, -10:]
        y = y[:, -10:]

        y_pred = torch.flatten(y_pred, start_dim=0, end_dim=1)
        y = torch.flatten(y, start_dim=0, end_dim=1)

        return self.multiclass_acc(y_pred, y)

"""
y_pred : [batch_size * length]
y      : [batch_size * length]

=> score / batch_size where score += 1 if entire sequence is correct
"""
class SequenceAccuracy():
    def __init__(self, nb_classes, device="cpu"):
        self.multiclass_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=nb_classes)

    def to(self, device):
        self.multiclass_acc.to(device)

    def __call__(self, y_pred, y):
        batch_size = y_pred.shape[0]
        correct = 0

        for batch_idx in range(batch_size):
            if 1. == self.multiclass_acc(y_pred[batch_idx], y[batch_idx]):
                correct += 1

        return correct / batch_size
    
class AverageCorrectLength():
    def __call__(self, y_pred, y):
        batch_size = y_pred.shape[0]
        tot_correct_length = 0

        for batch_idx in range(batch_size):
            if (y_pred[batch_idx] == y[batch_idx]).min():
                curr_correct_length = y_pred.shape[1]
            else:
                curr_correct_length = np.argmin(y_pred[batch_idx].cpu() == y[batch_idx].cpu())
            tot_correct_length += curr_correct_length

        return tot_correct_length / batch_size
    
class MinimalCorrectLength():
    def __call__(self, y_pred, y):
        min_correct_length = y_pred.shape[1]

        for batch_idx in range(y_pred.shape[0]):
            if (y_pred[batch_idx] == y[batch_idx]).min():
                curr_correct_length = y_pred.shape[1]
            else:
                curr_correct_length = np.argmin(y_pred[batch_idx].cpu() == y[batch_idx].cpu())
            if curr_correct_length < min_correct_length:
                min_correct_length = curr_correct_length

        return min_correct_length

"""
Transformer hooks
"""
def store_activations_hook(
    value,
    hook: HookPoint,  # noqa: ARG001
    store: List,
):
    store.append(value)
    return value

def to_numpy(tensor: torch.TensorType):
    return tensor.cpu().detach().numpy()

def tensor_to_str_list(tensor):
    assert len(tensor.shape) == 2

    res = []
    for i in range(tensor.shape[0]):
        curr_str = ""
        for j in range(tensor.shape[1]):
            curr_str += str(tensor[i, j].item())
        res.append(curr_str)
    
    return res

def get_model_paths(project, run_id):
    return list(map(lambda x: f"models/{project}/{x}", filter(lambda name: (str(run_id) in name) and ("pt" in name), os.listdir(f"models/{project}/"))))