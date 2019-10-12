import torch
from torch import nn
from torch.nn import functional as F

class Attention(nn.Module):
    r"""Global attention from Luong et. al."""
    
    def __init__(self, dim=None, score_type="dot"):
        
        super(Attention, self).__init__()
        
        if score_type in ("general", "concat") and dim is None:
            raise AssertionError("dimension 'dim' of the hidden vectors needs "
                                 "to be specified for parameter initializations"
                                 " when score_type provided is either 'general'"
                                 " or 'concat'.")
        elif score_type != "dot":
            raise AssertionError("score_type should either be dot or general or"
                                 " concat.")
        
        self.dim = dim
        self.score_type = score_type
        
        # initialize scoring parameters e.g. W_a and v_a for the score type
        self._init_scoring_params()
        
        # function for calculating attention scores
        if score_type in ("dot", "general"):
            self.scoring_func = torch.bmm
        else: # concat
            self.scoring_func = lambda t, s: self.scoring_concat(t, s).sum(dim=-1)
        
        # function for obtaining final context vector
        self.context_func = torch.bmm
        
    def _init_scoring_params(self):
        
        if self.score_type == "general":
            self.W_a = nn.Linear(self.dim, self.dim, bias=False)
        
        elif self.score_type == "concat":
            self.W_a = nn.Linear(self.dim*2, self.dim, bias=False)
            self.v_a = nn.Parameter(torch.Tensor(self.dim))
    
    def scoring_concat(self, trg_outs, src_outs):
        r"""For obtaining v_a^T tanh(W_a[h_t;h_s]) in vector form. Reduction 
        to be performed on this vector later for obtaining scaler scores when 
        required.
        """
        
        batch_size, trg_len, trg_hidden_dim = trg_outs.size()
        src_len, src_hidden_dim = src_outs.size(1), src_outs.size(2)
        
        # tile target outputs along seq_len (batch x trg_len*src_len x trg_dim)
        tiled_trg_outs = trg_outs.repeat_interleave(src_len, dim=1)

        # repeat source outputs along seq_len (batch x trg_len*src_len x src_dim)
        repeated_src_outs = src_outs.repeat(1, trg_len, 1)

        # concat target with source outputs 
        # (batch x trg_len*src_len x trg_dim+src_dim)
        concat_outs = torch.cat((tiled_trg_outs, repeated_src_outs), dim=-1)
        
        # After linear and non-linear transformations, reshape to 
        # (batch x trg_len x src_len x dim)
        score_vec = (self.v_a * torch.tanh(self.W_a(concat_outs))).view(
                                            batch_size, trg_len, src_len, -1)
        
        return score_vec
    
    def calc_score(self, trg_outs, src_outs):
        r"""Currently only batch_first tensors are supported.
        
        Obtain scores between trg_outs and src_outs with tensor shape 
        transformation from:
        
        (batch x trg_len x dim) * (batch x src_len x dim)
        
        to:
        
        (batch x trg_len x src_len) or (batch x trg_len x src_len x dim)
        """
        
        if self.score_type == "dot":
            # for obtaining h_t^T h_s
            score = self.scoring_func(trg_outs, src_outs.transpose(1,2))
        
        elif self.score_type == "general":
            # for obtaining h_t^T W_a h_s
            score = self.scoring_func(self.W_a(trg_outs), src_outs.transpose(1,2))
        
        else:
            # for obtaining v_a^T tanh(W_a[h_t;h_s])
            score = self.scoring_func(trg_outs, src_outs)
        
        return score
    
    def forward(self, trg_outs, src_outs):
        r"""Currently only batch_first tensors are supported."""
        
        # If trg_outs is not a 3D tensor, unsqueeze it along the 1st i.e. 
        # seq_len dimension
        if len(trg_outs.shape) != 3:
            # batch x hidden_dim -> batch x seq_len(=1) x hidden_dim
            trg_outs = trg_outs.unsqueeze(1)
        
        # get attention score in a tensor form of either 
        # (batch x trg_len x src_len) or (batch x trg_len x src_len x dim)
        attn_scores = self.calc_score(trg_outs, src_outs)
        
        # attention to be calculated on the tensor dim for src_len i.e. 2
        # (batch x trg_len x src_len) or (batch x trg_len x src_len x dim)
        attn_weights = F.softmax(attn_scores, dim=2)
        
        context_vec = self.context_func(attn_weights, src_outs)
        
        return context_vec, attn_weights