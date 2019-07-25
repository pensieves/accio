from torch import nn

from .. import get_embedding, get_dense_block

class MultiEmbedTransform(nn.Module):
    
    def __init__(self, sparse_emb_cols=[], dense_emb_cols=[], 
                 sparse_emb_trans_args=dict(hidden_size=50, output_size=50, num_layers=2, 
                                            non_linearity="ReLU", dropout="Dropout", 
                                            dropout_kwargs=dict(p=0.2), 
                                            terminal_non_linearity=False, 
                                            terminal_dropout=False),
                 dense_emb_trans_args=None, aggregation="concat"):
        
        r"""{sparse, dense}_emb_cols: a list of dict or a single dict of 
        parameters for each cols, the parameters required are emb_size (only 
        requirement for sparse embeddings), emb_dim, emb_weight and train.
        
        {sparse, dense}_emb_trans_args: dictionary or list of dictionaries of 
        arguments to be passed to the dense tranformation layer on top of 
        embedding layers. If single dictionary is passed, the same arguments 
        are used to transform all the embeddings. Otherwise if list of 
        dictionaries are passed then they should be same in number as the 
        number of multiple embeddings required."""

        super(MultiColEmbedTransform, self).__init__()
        
        for emb_cols, trans_args in ((sparse_emb_cols, sparse_emb_trans_args),
                                     (dense_emb_cols, dense_emb_trans_args)):
            
            if isinstance(trans_args, list) and len(emb_cols) != len(trans_args):
                raise AssertionError("Length of {} and {} should match.".format(
                                        emb_cols, trans_args))
                
        assert aggregation in (None, "concat", "sum", "mean", "max")
        # TODO: output size check for dense embedding size and the embedding 
        # transformation output size when the aggregation type is not concat.
        
        if isinstance(sparse_emb_cols, dict):
            sparse_emb_cols = [sparse_emb_cols]
        if isinstance(dense_emb_cols, dict):
            dense_emb_cols = [dense_emb_cols]
        
        self.sparse_emb = nn.ModuleList()
        self.sparse_emb_transform = nn.ModuleList()
        
        self.dense_emb = nn.ModuleList()
        self.dense_emb_transform = nn.ModuleList()
        
        for emb_col in sparse_emb_cols:
            emb = get_embedding(emb_type="one-hot", 
                                num_embeddings=emb_col["emb_size"])
            self.sparse_emb.append(emb)
        
        for emb_col in dense_emb_cols:
            emb = get_embedding(emb_type="distributional", 
                                num_embeddings=emb_col.get("emb_size"),
                                embedding_dim=emb_col.get("emb_dim"), 
                                emb_weight=emb_col.get("emb_weight"),
                                freeze=(not emb_col.get("train", False)))
            self.dense_emb.append(emb)
        
        tup_for_sparse_transformation = (self.sparse_emb_transform, 
                                         sparse_emb_trans_args, sparse_emb_cols, 
                                         self.sparse_emb)
        
        tup_for_dense_transformation = (self.dense_emb_transform, dense_emb_trans_args, 
                                        dense_emb_cols, self.dense_emb)
        
        for trans_list, trans_args, emb_cols, emb in (tup_for_sparse_transformation, 
                                                      tup_for_dense_transformation):

            if trans_args is not None:
                if isinstance(trans_args, list):
                    for i in range(len(trans_args)):
                        trans_list.append(get_dense_block(input_size=emb[i].weight.shape[1], 
                                                          **trans_args[i]))
                else:
                    for i in range(len(emb_cols)):
                        trans_list.append(get_dense_block(input_size=emb[i].weight.shape[1],
                                                          **trans_args))
    
    def forward(self, sparse_col_inp=None, dense_col_inp=None):
#         import pdb; pdb.set_trace()
        for col_inp, emb, emb_transform in ((sparse_col_inp, self.sparse_emb, 
                                             self.sparse_emb_transform), 
                                            (dense_col_inp, self.dense_emb, 
                                             self.dense_emb_transform)):
            
            if col_inp is not None and isinstance(col_inp, list):
                # apply embedding
                for i in range(len(col_inp)):
                    col_inp[i] = emb[i](col_inp[i])
                
                # apply dense transformation after embedding
                for i, transform in enumerate(emb_transform):
                    col_inp[i] = transform(col_inp[i])
        
        return sparse_col_inp, dense_col_inp

# aggregation