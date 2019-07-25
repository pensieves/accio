import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

# module not yet ready
def customize_inception_v3(model, num_classes=1000):

    if num_classes != 1000:
        model.fc = nn.Linear(2048, num_classes)
    return model

class inception_v3(nn.Module):
    
    def __init__(self, pretrained_path=None, aux_logits=True, transform_input=False, 
                 customize_func=customize_inception_v3, **kwargs):
        
        super(inception_v3, self).__init__()

        self.base_model = models.inception_v3(pretrained=False, {"aux_logits" : aux_logits,
                                                    "transform_input" : transform_input})
        if pretrained_path is not None:
            self.base_model.load_state_dict(torch.load(pretrained_path))
        
        self.base_model = customize_func(model=self.base_model, **kwargs)
    
    def forward(self, x, mode='train', **kwargs):
        
        if mode == 'train':
            x = self.base_model(x)
        
        # featurize has to be merged with forward function because
        # the torch.nn.DataParallel can only deal with forward method
        # for the time being.
        elif mode == 'featurize':
            x = self.featurize(x, **kwargs)
            
        return x
    
    def featurize(self, x, feature_module_depth=1):
        x = self.base_model.features(x)
        x = x.view(x.size(0), -1)
        for i in range(feature_module_depth):
            x = self.base_model.classifier[i](x)
        return x