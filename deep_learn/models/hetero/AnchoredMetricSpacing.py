from torch import nn

class AnchoredMetricSpacing(nn.Module):
    
    def __init__(self, anchor_encoder, entity_encoder=None, anchor_dense=None, 
                 entity_dense=None):
        
        r"""Network to return vectors in metric space of relative alignment of 
        entities about anchors
        
        Args:
            anchor_encoder: A featurizer to convert anchor samples to vectors.
            entity_encoder: A featurizer to convert entity samples to vectors. 
                The featurizer is required when anchor and entities are of 
                different kind, thereby requiring different encoders. Default: 
                None. i.e. anchor_encoder will be used for entities too.
            anchor_dense: Dense layers to be applied to anchor vectors. Default: None
            entity_dense: Dense layers to be applied to entity vectors. Default: None
        """
        
        super(AnchoredMetricSpacing, self).__init__()
        
        self.anchor_encoder = anchor_encoder
        self.entity_encoder = anchor_encoder if entity_encoder is None else \
                                entity_encoder
        self.anchor_dense = anchor_dense
        self.entity_dense = anchor_dense if entity_dense is None else entity_dense
        
    def forward(self, anchor, entities, anchor_kwargs={}, entity_kwargs={}):
        r"""When the intention is to just featurize anchor or entities, the other 
        argument can be passed as None."""

        anchor_enc = self.featurize_anchor(anchor, **anchor_kwargs)
        entities_enc = self.featurize_entities(entities, **entity_kwargs)

        return anchor_enc, entities_enc

    def featurize_anchor(self, anchor, **kwargs):
        anchor_enc = None

        if anchor is not None:
            anchor_enc = self.anchor_encoder(anchor, **kwargs)            
            if self.anchor_dense is not None:
                anchor_enc = self.anchor_dense(anchor_enc)

        return anchor_enc

    def featurize_entities(self, entities, **kwargs):
        entities_enc = None

        if entities is not None:

            if not isinstance(entities, (tuple, list)):
                # i.e. if there is only duplet comparison instead of triplet
                entities = [entities]

            entities_enc = [self.entity_encoder(i, **kwargs) for i in entities]
            if self.entity_dense is not None:
                entities_enc = [self.entity_dense(i) for i in entities_enc]

            if len(entities_enc) == 1:
                entities_enc = entities_enc[0]

        return entities_enc