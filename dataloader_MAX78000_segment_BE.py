import torchvision
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, random_split
import numpy as np
from PIL import Image
import ai8x

import torch
from torchvision.io import read_image
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
import numpy as np
import cv2
import os, sys
from os import listdir

'''Dataset Class Definition- here we define class for loading data- ***ONGOING***

'''
class Segmentation_MAX(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, mask_transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.mask_transform = mask_transform

        self.image_files = [f for f in sorted(os.listdir(images_dir)) if f.endswith('.tif')]
        self.mask_files = [f for f in sorted(os.listdir(masks_dir)) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_files[idx])

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        ### crop images, downscale, resize, and normalize in here as well

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask).squeeze(0)

        return image, mask
        

class ToLabel:
    def __init__(self, label_mappings):
        self.label_mapping = label_mappings

    def __call__(self, pic):
        label_array = np.array(pic, dtype= np.int32)
        for k, v in self.label_mapping.items():
            label_array[label_array == v] = k
        return torch.from_numpy(label_array).long() 
    

class Normalize:
    def __init__(self, arg):
        self.arg = arg
    def __call__(self, img):
        if self.arg == 'test':
            return img.sub(0.5).mul(256.).round().clamp(min=-128, max=127)
        if self.arg == 'train':
            return img.sub(0.5).mul(256.).round().clamp(min=-128, max=127).div(128.)
        else:
            sys.exit()

'''Dataloader function- here we create the function the instantiates the previous class, and we 
create and implement transforms and call ai8x.normalize()--- ***ONGOING***
'''

def get_BE_dataset(data, load_train=True, load_test=True, fold_ratio=1):
    (data_dir, args) = data
    path_data = "/Users/ashutoshraman/Documents/repos/Tearney_BE/raw_data/images/"
    path_masks = "/Users/ashutoshraman/Documents/repos/Tearney_BE/raw_data/annotations/"

    image_transforms = transforms.Compose([
    transforms.ToTensor(),
    #Normalize(arg='train'),
    ai8x.normalize(args=args),
    ai8x.fold(fold_ratio=fold_ratio),
    ])

    label_mapping = {0: 0, 1: 150, 2: 76}

    mask_transforms = transforms.Compose([
        ToLabel(label_mapping),
    ])

    dataset = Segmentation_MAX(path_data, path_masks, image_transforms, mask_transforms)
    
     # Creating data indices for training and validation splits:
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    if not load_train:
        test_dataset = None
    if not load_test:
        train_dataset = None

    return train_dataset, test_dataset




'''Dataset dictionary- here we create the dictionary that will be read by the ai8x-training train.py 
script, it will be the name of the dataset in the command terminal call and the call for the dataloader
function in here as well--- ***COMPLETE***
'''
datasets = [
    {
        'name': 'be_segmentation_data',
        'input': (3, 1024, 1024),       # (48, 88, 88): use this size eventually once cropped and downsampled and folded
        'output': (0, 1, 2),
        'weight' : (1, 1, 1),   # optional input to dict, just initializes weights, if value not given it just assumes (1,1) anyway
        'loader': get_BE_dataset,
    }
]