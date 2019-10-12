import numpy as np
import pandas as pd

from bisect import bisect_right

import torch
from torch.utils.data import Dataset

class WindowBasedSequenceDataset(Dataset):
    
    def __init__(self, seq, seq_ids, feat_cols=None, window_len=2, 
                 pos_reg_len_in_seq=2):
        
        self.seq = seq
        self.feat_cols = feat_cols
        
        if isinstance(seq_ids, str):
            self.seq_ids = seq[seq_ids].values
        else:
            self.seq_ids = np.array(seq_ids)
            
        self.window_len = window_len
        
        if isinstance(pos_reg_len_in_seq, str):
            raise NotImplementedError
#             pass # timestamp based, may differ for sequence
        else:
            self.pos_reg_len_in_seq = pos_reg_len_in_seq
        
        unique_seq_id, seq_start_idx = np.unique(self.seq_ids, return_index=True)
        sort_select_seq_start_idx =  np.argsort(seq_start_idx)
        
        self.seq_start_idx = np.append(seq_start_idx[sort_select_seq_start_idx], 
                                       len(self.seq_ids))
        self.unique_seq_id = unique_seq_id[sort_select_seq_start_idx]
        
        # inclusive for neg reg start
        self.seq_neg_start_idx = self.seq_start_idx[:-1] + self.window_len - 1

        # exclusive for neg reg end, inclusive for pos reg start
        self.seq_pos_start_idx = self.seq_start_idx[1:] - self.pos_reg_len_in_seq
        
        # inclusive for pos reg end
        self.seq_pos_end_idx = self.seq_start_idx[1:] - 1

        self.cumul_num_windows_in_seq = np.cumsum(self.seq_start_idx[1:] - 
                                                  self.seq_start_idx[:-1] - 
                                                  self.window_len + 1)
        
        # insert 0 as the beggining count for handy computation
        self.cumul_num_windows_in_seq = np.insert(self.cumul_num_windows_in_seq,
                                                  obj=0, values=0)
    
    def __len__(self):
        return self.cumul_num_windows_in_seq[-1]
        
    def __getitem__(self, i):
        
        if i >= len(self):
            raise IndexError("Index out of range for dataset of length {}."\
                             .format(len(self)))
            
        seq_id = bisect_right(self.cumul_num_windows_in_seq, i) - 1
        
        idx_within_seq_id = (i-self.cumul_num_windows_in_seq[seq_id])

        idx_in_seq = self.seq_neg_start_idx[seq_id] + idx_within_seq_id
        
        label = int(idx_in_seq >= self.seq_pos_start_idx[seq_id])

        seq_feats = self.seq if self.feat_cols is None else \
                        self.seq[self.feat_cols].values
        
        window = seq_feats[idx_in_seq-self.window_len+1:idx_in_seq+1]
        
        return window, seq_id, label