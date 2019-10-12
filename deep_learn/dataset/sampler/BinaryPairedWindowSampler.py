import numpy as np
from math import ceil

import torch
from torch.utils.data.sampler import Sampler

class BinaryPairedWindowSampler(Sampler):
    
    def __init__(self, cumul_num_windows_in_seq, seq_neg_start_idx, 
                 seq_pos_start_idx, stride=1, num_strided_sample=1, 
                 permute=True, seed=None):
        r"""cumul_num_windows_in_seq, seq_neg_start_idx, seq_pos_start_idx 
        are specific to dataset and should come from a dataset object."""
        
        self.cumul_num_windows_in_seq = cumul_num_windows_in_seq
        # -1 to account for 0 value at the start of cumul_num_windows_in_seq
        # which would have been inserted for handy computation.
        self.num_seq_ids = len(cumul_num_windows_in_seq) - 1
        
        # insert 0 in the beggining for handy computation, if not already there
        if self.cumul_num_windows_in_seq[0] != 0:
            self.cumul_num_windows_in_seq = \
                    np.insert(self.cumul_num_windows_in_seq, obj=0, values=0)
            self.num_seq_ids += 1
        
        self.seq_neg_start_idx = seq_neg_start_idx
        self.seq_pos_start_idx = seq_pos_start_idx
        
        self.stride = stride
        self.num_strided_sample = num_strided_sample
        
        self.permute = permute
        if permute is True and seed is not None:
            np.random.seed(seed)
        
        # segmented stride for starting groups of overlapping window strides
        self.seg_stride = (stride - 1) + (num_strided_sample - 1) * stride + 1
        
        self.paired_window_idx = tuple(self._circular_strided_paired_window_idx(i) 
                                       for i in range(self.num_seq_ids))
        
        self.paired_window_idx = np.concatenate(self.paired_window_idx, axis=0)
    
    def _circular_strided_window_idx(self, seq_idx, num_windows, 
                                     circ_num_windows, neg_reg=True):
        
        window_idx = None
        
        if num_windows > 0:
            
            if neg_reg is True:
                start_idx = self.cumul_num_windows_in_seq[seq_idx]
                end_idx = start_idx + num_windows
            
            else:
                end_idx = self.cumul_num_windows_in_seq[seq_idx+1]
                start_idx = end_idx - num_windows

            window_idx = np.arange(start_idx, end_idx)
            window_idx = np.take(window_idx, indices=range(circ_num_windows),
                                 mode="wrap")

            window_idx = window_idx.reshape(-1, self.stride).transpose()\
                                        .reshape(-1, self.num_strided_sample)
        
        return window_idx
    
    def _single_to_paired_reshape(self, window_idx):
        
        expanded_shape = (window_idx.shape[0] + 1 
                          if window_idx.shape[0] % 2 == 1 
                          else window_idx.shape[0])
        
        return np.take(window_idx, indices=range(expanded_shape), axis=0, 
                       mode="wrap").reshape(expanded_shape//2, -1)
    
    def _paired_window_idx(self, neg_window_idx, pos_window_idx):
        
        paired_window_idx = None
        
        if neg_window_idx is not None and pos_window_idx is not None:
            paired_window_idx = np.insert(neg_window_idx,
                                          obj=range(1,neg_window_idx.shape[1]+1), 
                                          values=pos_window_idx, axis=1)
        
        elif neg_window_idx is None and pos_window_idx is not None:
            paired_window_idx = self._single_to_paired_reshape(pos_window_idx)
        
        elif neg_window_idx is not None and pos_window_idx is None:
            paired_window_idx = self._single_to_paired_reshape(neg_window_idx)
        
        return paired_window_idx

    def _circular_strided_paired_window_idx(self, seq_idx):
        
        num_neg_windows = (self.seq_pos_start_idx[seq_idx] - 
                           self.seq_neg_start_idx[seq_idx])
        
        num_pos_windows = (self.cumul_num_windows_in_seq[seq_idx+1] - 
                           self.cumul_num_windows_in_seq[seq_idx] - 
                           num_neg_windows)

        circ_num_neg_windows = (ceil(num_neg_windows / self.seg_stride) * 
                                self.seg_stride)
        
        circ_num_pos_windows = (ceil(num_pos_windows / self.seg_stride) * 
                                self.seg_stride)

        circ_num_windows = max(circ_num_neg_windows, circ_num_pos_windows)
        
        neg_window_idx = self._circular_strided_window_idx(seq_idx, 
                                                           num_neg_windows, 
                                                           circ_num_windows, 
                                                           neg_reg=True)
        
        pos_window_idx = self._circular_strided_window_idx(seq_idx, 
                                                           num_pos_windows, 
                                                           circ_num_windows, 
                                                           neg_reg=False)
        
        paired_window_idx = self._paired_window_idx(neg_window_idx, 
                                                    pos_window_idx)
        
        return paired_window_idx
        
    def __len__(self):
        raise self.paired_window_idx.size
        
    def __iter__(self):
        if self.permute is True:
            return iter(np.random.permutation(self.paired_window_idx).ravel())
        return iter(self.paired_window_idx.ravel())