import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import transforms

resize_dict = {'vgg16' : (224,224), 'inception_v3' : (229,229)}

# Architectures where transforms pertaining to mean and stddev
# can be performed inside the architecture itself have their
# mean and stddev set as 0 and 1 respectively.
mean_dict = {'vgg16' : np.array([0.485, 0.456, 0.406]),
             'inception_v3' : np.zeros(3)}

std_dict = {'vgg16' : np.array([0.229, 0.224, 0.225]),
            'inception_v3' : np.ones(3)}

transforms_dict = {'vgg16' : transforms.Compose([
                                transforms.Resize(resize_dict['vgg16']),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=mean_dict['vgg16'],
                                                    std=std_dict['vgg16'])]),

                   'inception_v3' : transforms.Compose([
                                transforms.Resize(resize_dict['inception_v3']),
                                transforms.ToTensor()])
                   }

def tensor_to_image(tensor, mean = mean_dict['vgg16'],
                    std = std_dict['vgg16'], save_path=None):
    plt.figure()
    img = tensor.numpy().transpose((1, 2, 0))
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    if save_path is not None:
        plt.savefig(save_path)
    return img