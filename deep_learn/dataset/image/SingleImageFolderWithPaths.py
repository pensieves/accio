import os
from torch.utils.data import Dataset
from torchvision.datasets.folder import IMG_EXTENSIONS
from torchvision.datasets.folder import default_loader
from torchvision.datasets.folder import has_file_allowed_extension

class SingleImageFolderWithPaths(Dataset):
    def __init__(self, root, transform=None, loader=default_loader, 
                 extensions=IMG_EXTENSIONS, image_name_list=None,
                 file_count_limit=None):
        
        self.root = root
        self.loader = loader
        self.extensions = extensions
        
        self.file_count_limit = file_count_limit
        self.images = self.make_dataset(image_name_list)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        path = self.images[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, path

    def make_dataset(self, image_name_list=None):
        if image_name_list is not None:
            
            if self.file_count_limit is None:
                self.file_count_limit = len(image_name_list)
            
            images = [os.path.join(self.root, fname) for fname in \
                              image_name_list[:self.file_count_limit]]
        
        else:
            images = []
            root_dir = os.path.expanduser(self.root)

            for i, fname in enumerate(os.listdir(root_dir)):
                
                if has_file_allowed_extension(fname, self.extensions):
                    
                    if self.file_count_limit is not None and \
                                        i == self.file_count_limit:
                        break
                    
                    path = os.path.join(root_dir, fname)
                    images.append(path)
                    
        return images
        