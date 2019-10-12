from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ..utils import get_embedding

class EncoderRNN(nn.Module):
    
    def __init__(self, emb_type="distributional", num_embeddings=None, embedding_dim=None, 
                 emb_weight=None, freeze=True, emb_kwargs={}, sequencer="LSTM", input_size=None, 
                 hidden_size=100, num_layers=1, bias=True, batch_first=True, dropout=0, 
                 bidirectional=True):
        
        super(EncoderRNN, self).__init__()

        self.embedding = None
        if emb_type is not None:
            self.embedding = get_embedding(emb_type, num_embeddings, embedding_dim, emb_weight, 
                                           freeze, **emb_kwargs)
            input_size = self.embedding.weight.shape[1]
        
        self.batch_first = batch_first
        
        self.sequencer = nn.__dict__[sequencer](input_size, hidden_size, num_layers, bias, 
                                                batch_first, dropout, bidirectional)
    
    def forward(self, inp, inp_len=None):
        
        if self.embedding is not None:
            inp = self.embedding(inp)
            if inp_len is not None:
                total_length = inp.size(int(self.batch_first))
                
                # TODO: Even though length dimension has been identified using 
                # self.batch_first, it currently works only for batch_first=True.
                # In the opposite case, additional manipulation on tensor shapes
                # will be required to make it work with pack_padded_sequence or
                # pad_packed_sequence.
                # REF: https://pytorch.org/docs/stable/notes/faq.html#pack-rnn-unpack-with-data-parallelism
                
                inp = pack_padded_sequence(inp, inp_len, self.batch_first)
        
        out, hidden = self.sequencer(inp)
        if inp_len is not None:
            out, _ = pad_packed_sequence(out, self.batch_first, 
                                         total_length=total_length)
        
        return out, hidden