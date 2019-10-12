import torch
from torch import nn
from random import random
import numpy as np
from .EncoderRNN import EncoderRNN
from ..utils import get_dense_block, bi2uni_dir_rnn_hidden

# Attention = lambda x: x # just a placeholder, replace with attention module once ready.

class DecoderRNN(EncoderRNN):
    
    def __init__(self, init_token_id, eos_token_id, emb_type="distributional", 
                 num_embeddings=None, embedding_dim=None, emb_weight=None, freeze=True, 
                 emb_kwargs={}, enc_sequencer="LSTM", enc_bidirectional=True, 
                 enc_hidden_size=None, dec_init_hidden_transform=False, max_len=50, 
                 attention=False, linear_bias=True, sequencer="LSTM", input_size=None, 
                 hidden_size=100, num_layers=1, bias=True, batch_first=True, dropout=0):
        
        # State-based Decoder (i.e. RNN-based) is kept unidirectional.
        super(DecoderRNN, self).__init__(emb_type, num_embeddings, embedding_dim, 
                                         emb_weight, freeze, emb_kwargs, sequencer, 
                                         input_size, hidden_size, num_layers, bias, 
                                         batch_first, dropout, bidirectional=False)
        
        self.init_token_id = init_token_id
        self.eos_token_id = eos_token_id
        self.max_len = max_len
        
        self.enc2dec_hidden = bi2uni_dir_rnn_hidden if enc_bidirectional else (lambda x: x)
        
        # if encoder hidden size is not specified, it is inferred to be the same as 
        # decoder hidden size
        if enc_hidden_size is None:
            enc_hidden_size = hidden_size
        
        self.dec_init_hidden_transform = None
        if dec_init_hidden_transform is True:
            self.dec_init_hidden_transform = self._dec_init_hidden_transform(hidden_size, 
                                                        enc_sequencer, enc_bidirectional, 
                                                        enc_hidden_size, linear_bias)
        else:
            if (int(enc_bidirectional)+1)*enc_hidden_size != hidden_size:
                raise AssertionError("num_directions * encoder hidden size should be"
                                     " equal to the hidden size of decoder if "
                                     "dec_init_hidden_transform is not being applied.")
        
        self.attention = Attention(hidden_size) if attention else None
        
        output_size = num_embeddings if num_embeddings is not None else \
                        self.embedding.weight.shape[0]
        self.output = nn.Linear(hidden_size, output_size, linear_bias)
    
    def forward_step(self, inp, dec_hidden, enc_out):
        
        emb = self.embedding(inp)
        dec_out, dec_hidden = self.sequencer(emb, dec_hidden)
        
        attention = None
        if self.attention is not None:
            dec_out, attention = self.attention(dec_out, enc_out)
        
        import pdb; pdb.set_trace()
#         log_softmax = torch.log_softmax
    
    def forward(self, inp=None, enc_out=None, enc_hidden=None, teach_force_ratio=1, 
                device=torch.device("cpu")):
        
        inp, batch_size, max_len = self._infer_args(inp, enc_out, enc_hidden, 
                                                    teach_force_ratio, device)
        
        dec_hidden = self._init_hidden(enc_hidden)
        
        use_teach_force = random() < teach_force_ratio
        batch_lengths = np.full((batch_size,), max_len, dtype=int)
        
        if use_teach_force:
            dec_inp = inp[:,:-1]
            self.forward_step(dec_inp, dec_hidden, enc_out)
    
    def _dec_init_hidden_transform(self, hidden_size, enc_sequencer, enc_bidirectional, 
                                   enc_hidden_size, linear_bias):
        
        num_enc_directions = int(enc_bidirectional) + 1

        if enc_sequencer == "LSTM":
            transform = nn.ModuleList(nn.Linear(num_enc_directions*enc_hidden_size, 
                                                hidden_size, linear_bias) 
                                      for i in range(2))
        else:
            transform = nn.Linear(num_enc_directions*enc_hidden_size, hidden_size, 
                                  linear_bias)
        
        return transform
    
    def _infer_args(self, inp, enc_out, enc_hidden, teach_force_ratio, device):
        
        # attention requirement validation
        if self.attention is not None and enc_out is None:
            raise ValueError("Argument enc_out cannot be None when attention is to"
                             " be used.")
            
        # inferring batch size
        if inp is not None:
            batch_size = inp.shape[0]
        elif enc_hidden is not None:
            if isinstance(enc_hidden, tuple):
                batch_size = enc_hidden[0].shape[1]
            else:
                batch_size = enc_hidden.shape[1]
            # overwrite the device variable with the enc_hidden device if available
            device = enc_hidden.device
        else:
            batch_size = 1
        
        # get input and max decoding length
        if inp is None:
            if teach_force_ratio > 0:
                raise ValueError("Set teach_force_ratio to 0 when no inp is provided.")
            
            # TODO: implement direct vector based decoding in addition to id based
            # decoding.
            if self.init_token_id is None:
                raise ValueError("A pre-identified value for init_token_id is "
                                 "required.")
            
            inp = torch.full(size=(batch_size, 1), fill_value=self.init_token_id, 
                             dtype=torch.long, device=device)
            
            max_len = self.max_len
        
        else:
            max_len = inp.shape[1] - 1 # confirm if -1 is required to account for 
                                       # init_token_id
        
        return inp, batch_size, max_len
    
    def _init_hidden(self, enc_hidden=None, non_linearity="tanh"):
        r"""returns initial decoder state conditioned on the last encoder hidden 
        state"""
        
        dec_hidden_init = None
        
        if enc_hidden is not None:
            
            # if enc_sequencer is LSTM providing (h_n, c_n)
            if isinstance(enc_hidden, tuple):
                
                dec_hidden_init = tuple(self.enc2dec_hidden(i) for i in enc_hidden)
                
                # apply further transform if transform is not None
                if self.dec_init_hidden_transform is not None:
                    dec_hidden_init = tuple(transform(hidden) for transform, hidden
                                            in zip(self.dec_init_hidden_transform, 
                                                   dec_hidden_init))
                    
                    if non_linearity is not None:
                        dec_hidden_init = tuple(torch.__dict__[non_linearity](i) 
                                                for i in dec_hidden_init)
            
            # if enc_sequencer is GRU
            else:
                dec_hidden_init = self.enc2dec_hidden(enc_hidden)
                if self.dec_init_hidden_transform is not None:
                    dec_hidden_init = self.dec_init_hidden_transform(dec_hidden_init)
                    if non_linearity is not None:
                        dec_hidden_init = torch.__dict__[non_linearity](dec_hidden_init)
        
        return dec_hidden_init