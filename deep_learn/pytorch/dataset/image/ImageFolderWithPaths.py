import torch
from torchvision import datasets

class ImageFolderWithPaths(datasets.ImageFolder):
    """
    Modification of ImageFolder dataset to return image file paths along 
    with the images and targets.
    
    The requirements for parameters while initializations will be same as
    that for ImageFolder dataset class, the signature of which is provided
    below:
    
    torchvision.datasets.ImageFolder(root, transform=None, 
                target_transform=None, loader=<function default_loader>)
    """

    # override the __getitem__ method which is called by dataloader
    def __getitem__(self, index):
        # get image and targets from the parent class __getitem__ method
        img, target = super(ImageFolderWithPaths, self).__getitem__(index)
        # get the image file path
        path = self.imgs[index][0]
        return img, target, path