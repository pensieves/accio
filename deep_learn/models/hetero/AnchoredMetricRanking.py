import torch
from torch import nn
from accio.deep_learn.models.hetero.AnchoredMetricSpacing import AnchoredMetricSpacing
# from accio.deep_learn import models
from ... import models

_anchor_encoder_creator = {"type": "image", "class": "vgg16", "kwargs": dict()}
_anchor_entity_combine = lambda x,y: torch.mul(x,y)
_anchor_entity_dense = lambda x: torch.sum(x, dim=1)

class AnchoredMetricRanking(AnchoredMetricSpacing):
    
    def __init__(self, anchor_encoder, entity_encoder=None, anchor_dense=None, 
                 entity_dense=None, anchor_entity_combine=_anchor_entity_combine, 
                 anchor_entity_dense=_anchor_entity_dense):
        
        r"""Network to return metric e.g. distance or similarity scores for entities 
        with respect to anchors. 
        
        Args:
            anchor_encoder: A featurizer to convert anchor samples to vectors.
            entity_encoder: A featurizer to convert entity samples to vectors. 
                The featurizer is required when anchor and entities are of 
                different kind, thereby requiring different encoders. Default: 
                None. i.e. anchor_encoder will be used for entities too.
            anchor_dense: Dense layers to be applied to anchor vectors. Default: None
            entity_dense: Dense layers to be applied to entity vectors. Default: None
            anchor_entity_combine - Function defining how anchors and entities are
                combined in the metric space. Default: Element-wise multiplication.
            anchor_entity_dense - Dense layers / function to be applied on top of the 
                anchor entity metric combination. Default: Sum of elements.
        
        Anchor and Entity dense transformations are to be such that the anchor entity 
        combine function can be applied based on the vector sizes of anchor and 
        entities after transformations."""
        
        super(AnchoredMetricRanking, self).__init__(anchor_encoder, entity_encoder, 
                                                    anchor_dense, entity_dense)
        
        self.anchor_entity_combine = anchor_entity_combine
        self.anchor_entity_dense = anchor_entity_dense
    
    @classmethod
    def from_template(cls, anchor_encoder_creator=_anchor_encoder_creator, 
                      entity_encoder_creator=None, anchor_dense=None, 
                      entity_dense=None, 
                      anchor_entity_combine=_anchor_entity_combine, 
                      anchor_entity_dense=_anchor_entity_dense):
        
        anchor_encoder = models.__dict__[anchor_encoder_creator["type"]]\
                                .__dict__[anchor_encoder_creator["class"]](
                                        **anchor_encoder_creator["kwargs"])
        
        if entity_encoder_creator is None:
            entity_encoder = anchor_encoder
        
        else:
            entity_encoder = models.__dict__[entity_encoder_creator["type"]]\
                                    .__dict__[entity_encoder_creator["class"]](
                                            **entity_encoder_creator["kwargs"])
        
        return cls(anchor_encoder, entity_encoder, anchor_dense, entity_dense, 
                   anchor_entity_combine, anchor_entity_dense)
    
    def forward(self, anchor, entities, mode="train", anchor_kwargs={},
                entity_kwargs={}):
        r"""When the intention is to just featurize anchor or entities, the other 
        argument can be passed as None."""
        
        assert mode in ("train", "featurize")
        
        anchor_enc, entities_enc = super(AnchoredMetricRanking, 
                                         self).forward(anchor, entities, 
                                                       anchor_kwargs, entity_kwargs)
            
        anchor_entities_enc = self.combine_anchor_entities(anchor_enc, 
                                                           entities_enc)
        
        anchor_entities_dense = self.apply_dense_on_anchor_entities_enc(
                                                        anchor_entities_enc)
        
        if mode == "featurize":
            return anchor_enc, entities_enc
        return anchor_entities_dense
    
    def combine_anchor_entities(self, anchor_enc, entities_enc):
        anchor_entities_enc = None
        
        if anchor_enc is not None and entities_enc is not None:
            
            if not isinstance(entities_enc, (list, tuple)):
                entities_enc = [entities_enc]
            
            anchor_entities_enc = [self.anchor_entity_combine(anchor_enc, i)
                                        for i in entities_enc]
            
            if len(anchor_entities_enc) == 1:
                anchor_entities_enc = anchor_entities_enc[0]
        
        return anchor_entities_enc
    
    def apply_dense_on_anchor_entities_enc(self, anchor_entities_enc):
        anchor_entities_dense = None
        
        if self.anchor_entity_dense is None:
            anchor_entities_dense = anchor_entities_enc
            
        elif anchor_entities_enc is not None:
            
            if not isinstance(anchor_entities_enc, (list, tuple)):
                anchor_entities_enc = [anchor_entities_enc]
            
            anchor_entities_dense = [self.anchor_entity_dense(i) 
                                     for i in anchor_entities_enc]
            
            if len(anchor_entities_dense) == 1:
                anchor_entities_dense = anchor_entities_dense[0]
        
        return anchor_entities_dense