"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch.utils.data as data
import os.path
import torch.utils.data as data
from PIL import Image
import os
import os.path
import torchvision.transforms as tf

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG','.nil',',nil.gz',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def default_loader(path):
    # img1 = PIL.Image.open(path1)
    img = Image.open(path)
    img = tf.Resize((256, 256))(img)
    return img
    

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = sorted(make_dataset(root))
        self.root = root
        # print(self.root)
        self.imgs = imgs
        # print(len(self.imgs))
        self.transform = transform
        self.return_paths = return_paths
        # print(self.return_paths)
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)