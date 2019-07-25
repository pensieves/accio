from torchtext.vocab import Vocab, Vectors
from collections import Counter
from copy import deepcopy

def vocab_from_vectors(vector_kwargs_list, vocab_kwargs):
    r"""Get Vocab object encompassing all the words in each vector list items.
    Each item in vector_kwargs_list corresponds to the kwargs needed to obtain
    individual vector lookup table. All of these are combined by concatenation
    to get a unified vocab object.
    
    NOTE: Since multiple vectors can be used, vector_kwargs_list must contain
    argument names even for positional arguments. Incase of vocab_kwargs, counter 
    and vectors will be inferred and hence need not be provided."""
    
    assert len(vector_kwargs_list) > 0
    vocab_kwargs = deepcopy(vocab_kwargs)
    
    # obtain vectors and counter from list of vector creating keyword arguments
    vectors = list()
    vocab_kwargs["counter"] = Counter()
    
    for kwargs in vector_kwargs_list:
        vecs = Vectors(**kwargs)
        vectors.append(vecs)
        vocab_kwargs["counter"].update(vecs.itos)
    
    vocab_kwargs["vectors"] = vectors
    vocab = Vocab(**vocab_kwargs)

    return vocab

special_tokens = {"init_token": "<start>", "eos_token": "<end>",
                  "pad_token": "<pad>", "unk_token": "<unk>"}