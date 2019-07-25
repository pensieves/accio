import torch
import torch.nn as nn

class MeanFeaturizer(nn.Module):
    def __init__(self):
        super(MeanFeaturizer, self).__init__()
        self.featurizer = torch.mean
        
    def forward(self, x, mode='train'):

        if mode == 'train':
            x = self.featurizer(x, dim=1)
        
        # featurize has to be merged with forward function because
        # the torch.nn.DataParallel can only deal with forward method
        # for the time being.
        elif mode == 'featurize':
            x = self.featurize(x)
            
        return x
    
    def featurize(self, x):
        return self.featurizer(x, dim=1)