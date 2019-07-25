import numpy as np
import torch

def MultiLabelMarginLossify(target):
    r"""
    Converts a numpy array or pytorch tensor of true multilabel with binary 
    indicator into the one required by torch.nn.MultiLabelMarginLoss i.e.
    the target consists of contiguous block of non-negative targets that 
    starts at the front.
    """
    
    if isinstance(target, torch.Tensor):
        target = target.data.numpy()
    new_target = np.full(target.shape,-1)
    
    row, col = target.nonzero()
    _, nonzero_col_count = np.unique(row, return_counts=True)
    new_col = np.concatenate([range(i) for i in nonzero_col_count])
    new_target[row, new_col] = col
    
    return torch.from_numpy(new_target)

def MultiLabelUnMarginLossify(target):
    r"""
    Converts a numpy array or pytorch tensor of the form required by 
    torch.nn.MultiLabelMarginLoss into multilabel with binary indicator.
    """
    
    if isinstance(target, torch.Tensor):
        target = target.data.numpy()
    new_target = np.zeros(target.shape, dtype='float32')
    
    row, _ = np.where(target >= 0)
    target = target[target != -1]
    new_target[row, target] = 1
    
    return torch.from_numpy(new_target)