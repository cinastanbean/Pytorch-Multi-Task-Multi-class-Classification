#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 14:52:43 2018

Python 3.6.6 |Anaconda, Inc.| (default, Jun 28 2018, 11:07:29)

@author: pilgrim.bin@163.com
"""

import os
import os.path
import copy
import math
import random

import torch.utils.data as data

from PIL import Image
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

'''-------------------------------------------------------------------'''

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

'''-------------------------------------------------------------------'''

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images

# return all type filepath of this path
def get_filelist(path):
    filelist = []
    for root,dirs,filenames in os.walk(path):
        for fn in filenames:
            filelist.append(os.path.join(root,fn))
    return filelist

# return img filepath of this path
def get_img_filelist(path):
    IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
    filelist = []
    for root,dirs,filenames in os.walk(path):
        for fn in filenames:
            if has_file_allowed_extension(fn, IMG_EXTENSIONS):
                filelist.append(os.path.join(root,fn))
    return filelist


'''-------------------------------------------------------------------'''

class DatasetFolderParsing():
    """
    Args:
        root (string): Root directory path.
        extensions (list[string]): A list of allowed extensions.
        
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
    """

    def __init__(self, root):
        classes, class_to_idx = find_classes(root)
        print('---------------------------------')
        print("DatasetFolderParsing::class_to_idx = {}".format(class_to_idx))
        extensions = IMG_EXTENSIONS
        class_to_samples = {key : get_img_filelist(os.path.join(root, key)) for key in class_to_idx}
        
        for key in class_to_samples.keys():
            if not len(class_to_samples[key]) > 0:
                raise(RuntimeError("Found 0 files in subfolders of: " + root + "\{}".format(key) + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.extensions = extensions
        
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.class_to_samples = class_to_samples



class ImageFolder_SpecifiedNumber(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        dbparser : DatasetFolderParsing instance.
        number_dict : specified number of each dict.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(self, dbparser, number_dict=None,
                 transform=None, target_transform=None,
                 loader=default_loader):
        '''
        if not isinstance(dbparser, DatasetFolderParsing):
            raise(RuntimeError("dbparser must be DatasetFolderParsing instance!"))
        '''
        
        if number_dict is None:
            raise(RuntimeError("number_dict cannot set as None!"))
            
        # check if keys right
        for key in number_dict.keys():
            if not key in dbparser.classes:
                raise(RuntimeError("Unknown class = {}.".format(key)))
        
        super(ImageFolder_SpecifiedNumber, self).__init__()

        self.root = dbparser.root
        self.loader = loader
        self.extensions = dbparser.extensions

        self.classes = dbparser.classes
        self.class_to_idx = dbparser.class_to_idx
        
        samples = []
        print("ImageFolder_SpecifiedNumber:")
        for key in number_dict.keys():
            number = number_dict[key]
            this_class_samples = dbparser.class_to_samples[key]
            random.shuffle(this_class_samples)
            # 可改，改为队列式，同时保持随机性。改为每次向队列请求n个样本，队列根据自身长度返回相应的样本，
            # 尽可能保证外发样本的随机性是不够的，还要保证每一个样本参与训练的随机性是平均分布，而不是高斯分布。
            n_repeat = int(math.ceil(1. * number / len(this_class_samples)))
            sub_samples = []
            for i in range(n_repeat):
                sub_samples += this_class_samples
            sub_samples = sub_samples[:number]
            sub_samples = [(s, self.class_to_idx[key]) for s in sub_samples]
            samples += sub_samples
            print("class = {}, n_repeat = {}".format(key, n_repeat))
        
        random.shuffle(samples)
        random.shuffle(samples)
        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        #sample = path # for test
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

'''-------------------------------------------------------------------'''

class BaseNumberSpecifier():
    """
    Args:
        class_to_number_dict (dict): Dict with items (class_name, number).
        base_number (int): base numner of each class.
        
     Attributes:
        class_to_number_dict (dict): Dict with items (class_name, number).
    """

    def __init__(self, class_to_number_dict, base_number):
        
        if not len(class_to_number_dict) > 0:
            raise(RuntimeError("Error: len(class_to_number_dict) <= 0."))
        if not base_number > 0:
            raise(RuntimeError("Error: base_number > 0."))

        self.class_number = len(class_to_number_dict) 
        self.base_number = base_number
        self.class_to_number_dict = copy.deepcopy(class_to_number_dict)
        self.class_to_prec_dict = None
        for key in self.class_to_number_dict.keys():
            self.class_to_number_dict[key] = base_number
        
    def update(self, class_to_prec_dict):
        if not len(class_to_prec_dict) == len(self.class_to_number_dict):
            raise(RuntimeError("Error: len(class_to_prec_dict) != len(self.class_to_number_dict)."))
        self.class_to_prec_dict = copy.deepcopy(class_to_prec_dict)
        
        for key in class_to_prec_dict.keys():
            '''
            if class_to_prec_dict[key] < 0 or class_to_prec_dict[key] > 1: # =1 ?
                raise(RuntimeError("Error: class_to_prec_dict[key] < 0 or class_to_prec_dict[key] > 1"))
            '''
            # dummy protect
            if class_to_prec_dict[key] <= 0:
                class_to_prec_dict[key] = 0.001
            if class_to_prec_dict[key] >= 1:
                class_to_prec_dict[key] = 0.9999
        
        # 惩戒太过了
        '''
        #weight_dict = {key : - math.log(class_to_prec_dict[key]) for key in class_to_prec_dict.keys()}
        #weight_dict = {key : 1 - (class_to_prec_dict[key]) for key in class_to_prec_dict.keys()}

        sum_weight = sum([weight_dict[key] for key in weight_dict.keys()])
        number_ratio = float(self.class_number * self.base_number) / sum_weight
        for key in weight_dict.keys():
            self.class_to_number_dict[key] = int(number_ratio * weight_dict[key])
        '''
        
        # w = 1 - p
        # w = (1 - p)**2
        # K = M + (N - M) * w
        weight_dict = {key : 1 - (class_to_prec_dict[key]) for key in class_to_prec_dict.keys()}
        # weight_dict = {key : (1 - (class_to_prec_dict[key]))**2 for key in class_to_prec_dict.keys()}
        M = self.base_number
        N = M * 2 # raw = 3
        for key in weight_dict.keys():
            self.class_to_number_dict[key] = int(M + weight_dict[key] * (N - M))
        
if __name__ == '__main__':
    
    path = 'data'
    base_number = 5
    
    dbparser = DatasetFolderParsing(path)
    
    print('--------------base test----------------')
    number_dict = {key : 0 for key in dbparser.class_to_idx.keys()}
    base_number_specifier = BaseNumberSpecifier(number_dict, base_number)
    number_dict=base_number_specifier.class_to_number_dict
    dataloader = ImageFolder_SpecifiedNumber(dbparser,
                    number_dict=number_dict)
    print("number_dict = {}".format(number_dict))
    for d, target in dataloader:
        print("t-d = {}-{}".format(target, d))


    print('--------------fake p test----------------')
    fake_p = {key : random.random() for key in dbparser.class_to_idx.keys()}
    print("fake_p = {}".format(fake_p))
    base_number_specifier.update(fake_p)
    number_dict=base_number_specifier.class_to_number_dict
    dataloader = ImageFolder_SpecifiedNumber(dbparser,
                    number_dict=number_dict)
    print("number_dict = {}".format(number_dict))
    
    for d, target in dataloader:
        print("t-d = {}-{}".format(target, d))
        
    