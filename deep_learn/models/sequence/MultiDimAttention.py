from . import Attention

class MultiDimAttention(Attention):
    r"""Multi-Dimensional Global Attention where attention is applied to each 
    dimension of the source output vectors."""
    
    def __init__(self, dim=None, score_type="dot"):
        
        super(MultiDimAttention, self).__init__(dim, score_type)
        
        # function for calculating attention scores
        if score_type in ("dot", "general"):
            # equivalent to torch.einsum("bik,bjk->bijk", t, s)
            # (batch x trg_len x dim), (batch x src_len x dim)
            #                    --> (batch x trg_len x src_len x dim)
            self.scoring_func = lambda t, s: t.unsqueeze(2) * s.unsqueeze(1)
        else: # concat
            self.scoring_func = self.scoring_concat
        
        # function for obtaining final context vector
        # (batch x trg_len x src_len x dim), (batch x src_len x dim)
        #                        --> (batch x trg_len x dim)
        self.context_func = lambda w, s: (w*s.unsqueeze(1)).sum(dim=2)