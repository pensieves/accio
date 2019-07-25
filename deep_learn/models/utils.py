import torch
from torch import nn

def get_embedding(emb_type="distributional", num_embeddings=None, embedding_dim=None, 
                  emb_weight=None, freeze=True, **kwargs):
    r"""Function to return embedding after validating the required arguments 
    for Embedding class.
    
    emb_type should either be one-hot or distributional."""
    
    assert emb_type in ("distributional", "one-hot")
    
    if emb_type == "one-hot" and num_embeddings is not None:
        embedding_dim = num_embeddings
        emb_weight = torch.eye(num_embeddings)
    
    elif emb_type == "distributional" and emb_weight is not None:
        num_embeddings, embedding_dim = emb_weight.shape
        
    if num_embeddings is None or embedding_dim is None:
        raise AssertionError("num_embeddings and embedding_dim should not be None. Either "
                             "provide their values explicity, or provide the emb_weight "
                             "value to infer their values or provide at least num_embeddings"
                             " in case of one-hot emb_type.")
    
    embedding = nn.Embedding(num_embeddings, embedding_dim, _weight=emb_weight, **kwargs)
    embedding.weight.requires_grad = not freeze
    
    return embedding

def get_dense_block(input_size, hidden_size=None, output_size=1, bias=True, num_layers=1, 
                     non_linearity="ReLU", non_linear_kwargs={}, dropout="Dropout", 
                     dropout_kwargs={}, terminal_non_linearity=False, terminal_dropout=False):
    r"""Function to construct and return a block of multiple linear layers along 
    with non-linearity and dropout applied to them."""
    
    assert (num_layers == 1 and hidden_size is None) or \
            (num_layers > 1 and hidden_size is not None)
    
    block = []
    
    if num_layers == 1:
        # if num_layers is 1 and no non-linearity and dropout is to be applied
        # then directly using nn.Linear would be straightforward. However, the 
        # case dealing with such a scenario is dealt with for the sake of 
        # completeness.
        block.append(nn.Linear(input_size, output_size, bias))
        
    else:
        block.append(nn.Sequential(nn.Linear(input_size, hidden_size, bias), 
                                   nn.__dict__[non_linearity](**non_linear_kwargs),
                                   nn.__dict__[dropout](**dropout_kwargs)))
        
        block.extend([nn.Sequential(nn.Linear(hidden_size, hidden_size, bias), 
                                    nn.__dict__[non_linearity](**non_linear_kwargs),
                                    nn.__dict__[dropout](**dropout_kwargs))
                      for i in range(num_layers-2)])
        
        block.append(nn.Linear(hidden_size, output_size, bias))
    
    if terminal_non_linearity is True:
        block.append(nn.__dict__[non_linearity](**non_linear_kwargs))
    
    if terminal_dropout is True:
        block.append(nn.__dict__[dropout](**dropout_kwargs))
    
    if len(block) == 1:
        return block[0]
    return nn.Sequential(*block)

def bi2uni_dir_rnn_hidden(hidden):
    num_layers_and_dir, batch, hidden_size = hidden.shape
    # since input is bidirectional and needs to be converted to unidirection
    num_directions = 2 
    num_layers = num_layers_and_dir // num_directions

    # hidden needs to be first split across 1st dimension to separate out 
    # num_layers and num_directions in that order, then it needs to be 
    # permuted to bring num_directions just before hidden_size which will 
    # then be combined to get a unidirectional hidden tensor :
    # (num_layers * num_directions, batch, hidden_size) ->
    # (num_layers, num_directions, batch, hidden_size) ->
    # (num_layers, batch, num_directions * hidden_size)
    # REF: shape details of h_n and c_n in the Outputs section of LSTM at
    # https://pytorch.org/docs/stable/nn.html#torch.nn.LSTM
    hidden = hidden.view(num_layers, num_directions, batch, hidden_size)
    hidden = hidden.permute(0,2,1,3).contiguous().view(num_layers, batch, -1)
    
    # another popular means to achieve this is by manual concatenation as:
    # hidden_fwd = hidden[0:hidden.shape[0]:2]
    # hidden_bwd = hidden[1:hidden.shape[0]:2]
    # hidden = torch.cat([hidden_fwd, hidden_bwd], dim=2)

    return hidden