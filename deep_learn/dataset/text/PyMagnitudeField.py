import torch
from torchtext.data import Field
from pymagnitude import Magnitude

class PyMagnitudeField(Field):

    def __init__(self, magnitude_vector_filepath, sequential=True, 
                 lower=True, tokenize=(lambda s: s.split()), 
                 include_lengths=True, batch_first=True, **kwargs):
        
        if kwargs.get('use_vocab'):
            kwargs['use_vocab'] = False
        self.vectors = Magnitude(magnitude_vector_filepath)
        super(PyMagnitudeField, self).__init__(sequential=sequential,
                                             lower=lower, 
                                             tokenize=tokenize, 
                                             include_lengths=include_lengths, 
                                             batch_first=batch_first, 
                                             **kwargs)
    
    def build_vocab(self, *args, **kwargs):
        pass
        
    def process(self, batch, device, train):
        if self.include_lengths:
            batch = (batch, [len(x) for x in batch])
        return self.numericalize(batch, device=device, train=train)
    
    def numericalize(self, arr, device=torch.device('cpu'), train=True):
        
        if self.include_lengths and not isinstance(arr, tuple):
            raise ValueError('Field has include_lengths set to True, but '
                             'input data is not a tuple of '
                             '(data batch, batch lengths).')
        
        if isinstance(arr, tuple):
            arr, lengths = arr
            lengths = torch.LongTensor(lengths)
        
        arr = torch.from_numpy(self.vectors.query(arr))
        if self.sequential and not self.batch_first:
            arr.t_()
           
        if device.type == 'cpu':
            if self.sequential:
                arr = arr.contiguous()
        else:
            arr = arr.cuda(device)
            if self.include_lengths:
                lengths = lengths.cuda(device)
        
        arr.requires_grad = False
        if self.include_lengths:
            return arr, lengths
        return arr
