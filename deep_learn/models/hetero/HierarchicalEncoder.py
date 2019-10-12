import torch
from torch import nn

import copy

from ... import models

def sequencer_out_featurizer(seq_out, seq_class="LSTM", bidirectional=False, 
                             all_layers=False):
    
    # seq_out is output, (h_n, c_n) for LSTM and is output, h_n for GRU
    out_feat = seq_out[1][0] if seq_class == "LSTM" else seq_out[1]
    
    # if all layers is false extract features only from last layer
    # if bidirectional extract from last two positions of first tensor 
    # dimension else extract only from last position.
    start_idx = 0 if all_layers else -(int(bidirectional) + 1)
    
    # h_n is of shape (num_layers * num_directions x batch x hidden_size)
    batch_size = out_feat.shape[1]
    
    # (num_layers * num_directions x batch x hidden_size) -->
    #                (batch x num_layers * num_directions * hidden_size)
    out_feat = out_feat[start_idx:].permute(1,0,2).reshape(batch_size,-1)
    
    return out_feat

def get_constructor_featurizer_kwargs(creator):
    
    featurizer = None
    featurizer_kwargs = dict()
    
    if creator is not None:
            
            if creator["type"] == "function":
                
                featurizer = _featurizer_funcs[creator["class"]]
                featurizer_kwargs = copy.deepcopy(creator["kwargs"])
            
            else:
                
                featurizer = models.__dict__[creator["type"]]\
                                    .__dict__[creator["class"]](
                                                **creator["kwargs"])
    return featurizer, featurizer_kwargs
    
class HierarchicalEncoder(nn.Module):
    
    _featurizer_funcs = {"sequencer_out_featurizer": sequencer_out_featurizer}

    _default_encoder_creator = {"type": "sequence", "class": "EncoderRNN", 
                                "kwargs": dict()}

    _default_featurizer_creator = {"type": "function", "class": "sequencer_out_featurizer", 
                                   "kwargs": dict(seq_class="LSTM", bidirectional=False, 
                                                  all_layers=False)}
    
    def __init__(self, inner_encoder, outer_encoder, inner_featurizer=None, 
                 inner_featurizer_kwargs=dict(), outer_featurizer=None, 
                 outer_featurizer_kwargs=dict()):
        r"""Meta-network architecture to perform hierarchical encoding for 
        samples of the form (batch x outer_size x inner_size x *), where
        outer entities are composed of inner entities which in turn may have 
        learnable or fixed representations.
        
        Args:
            inner_encoder: An encoder for inner entities (e.g. words).
            outer_encoder: An encoder for outer entities (e.g. sentences).
        
        NOTE: Network currently doesn't supports padding or packing unless the
        encoders themselves are written in a way to handle them.
        """
        
        super(HierarchicalEncoder, self).__init__()
        
        self.inner_encoder = inner_encoder
        self.outer_encoder = outer_encoder
        
        self.inner_featurizer = inner_featurizer
        self.inner_featurizer_kwargs = inner_featurizer_kwargs
        
        self.outer_featurizer = outer_featurizer
        self.outer_featurizer_kwargs = outer_featurizer_kwargs
        
    @classmethod
    def from_template(cls, inner_encoder_creator=_default_encoder_creator, 
                      outer_encoder_creator=_default_encoder_creator, 
                      inner_featurizer_creator=_default_featurizer_creator, 
                      outer_featurizer_creator=_default_featurizer_creator):
        
        inner_encoder = models.__dict__[inner_encoder_creator["type"]]\
                                .__dict__[inner_encoder_creator["class"]](
                                        **inner_encoder_creator["kwargs"])
        
        outer_encoder = models.__dict__[outer_encoder_creator["type"]]\
                                .__dict__[outer_encoder_creator["class"]](
                                        **outer_encoder_creator["kwargs"])

#         inner_encoder = EncoderRNN(**inner_encoder_creator)
#         outer_encoder = EncoderRNN(**outer_encoder_creator)
        
        inner_featurizer, inner_featurizer_kwargs = \
                    get_constructor_featurizer_kwargs(inner_featurizer_creator)
        
        outer_featurizer, outer_featurizer_kwargs = \
                    get_constructor_featurizer_kwargs(outer_featurizer_creator)
        
        return cls(inner_encoder, outer_encoder, inner_featurizer, 
                   inner_featurizer_kwargs, outer_featurizer, 
                   outer_featurizer_kwargs)
    
    def forward(self, inp):
        
        batch_size = inp.shape[0]
        # (batch x outer_size x inner_size x *) --> 
        #                        (batch * outer_size x inner_size x *)
        inp = inp.view(-1, *inp.shape[2:])
        
        intermediate = self.inner_encoder(inp)
        
        if self.inner_featurizer is not None:
            if isinstance(self.inner_featurizer, nn.Module):
                intermediate = self.inner_featurizer(inp)
            else:
                intermediate = self.inner_featurizer(inp, 
                                            **self.inner_featurizer_kwargs)
        
        # (batch * outer_size x feat) --> (batch x outer_size x *)
        intermediate = intermediate.view(batch_size, -1, *intermediate.shape[1:])
        
        out = self.outer_encoder(intermediate)
        
        if self.outer_featurizer is not None:
            if isinstance(self.outer_featurizer, nn.Module):
                out = self.outer_featurizer(out)
            else:
                out = self.outer_featurizer(out, **self.outer_featurizer_kwargs)
                
        return out