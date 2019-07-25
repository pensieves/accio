import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

def customize_vgg(model, num_classes=1000, tune_feature_layers=True, 
                  classifier_pop_count=0):
        
    if tune_feature_layers is False:
        for p in model.features.parameters():
            p.requires_grad = False

    if num_classes != 1000 or classifier_pop_count != 0:
        if num_classes != 1000 and classifier_pop_count == 0:
            classifier_pop_count = 1

        # model.classifier has only 7 cjildren so ignore anything 
        # corresponding to classifier_pop_count > 7
        in_features = 25088 if classifier_pop_count >= 7 else 4096  

        classifier_modules = list(model.classifier.children())
        for i in range(min(classifier_pop_count,7)):
            classifier_modules.pop()

        classifier_modules.append(nn.Linear(in_features=in_features, 
                                            out_features=num_classes, 
                                            bias=True))

        model.classifier = nn.Sequential(*classifier_modules)

    return model

class vgg16(nn.Module):
    
    def __init__(self, pretrained_path=None, customize_func=customize_vgg,
                 **kwargs):
        
        super(vgg16, self).__init__()

        self.base_model = models.vgg16(pretrained=False)
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