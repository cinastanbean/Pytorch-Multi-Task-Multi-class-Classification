#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 14:52:43 2018

Python 3.6.6 |Anaconda, Inc.| (default, Jun 28 2018, 11:07:29)

@author: pilgrim.bin@163.com
"""

import sys
import os
import argparse
import copy
import random

import torch
#import torch.nn as nn
#import torch.nn.parallel
#import torch.backends.cudnn as cudnn
#import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
#import torchvision.models as models

# baiqi diy
import diy_folder as diy_folder

'''
MLDataloader load MTMC dataset as following directory tree.
Make sure train-val directory tree in consistance.

data_root_path
├── task_A
│   ├── train
│   │   ├── class_1
│   │   ├── class_2
│   │   ├── class_3
│   │   └── class_4
│   └── val
│       ├── class_1
│       ├── class_2
│       ├── class_3
│       └── class_4
└── task_B
    ├── train
    │   ├── class_1
    │   ├── class_2
    │   └── class_3
    └── val
        ├── class_1
        ├── class_2
        └── class_3
'''

def raise_error_if_not_exists(path):
    if not os.path.exists(path):
        raise(RuntimeError("Dataset path = {} cannot be found.".format(path)))

# Single Label Multi-Class Dataloader
class SLMCDataloader():
    def __init__(self, path, dataresize=[256,256,224,224], batch_size=32, workers=4, base_number=100):
        # input params
        self.batch_size = batch_size
        self.workers = workers
        self.base_number = base_number
        
        # directory_check
        self.rootpath = path
        self.train_path = os.path.join(path, 'train')
        self.val_path = os.path.join(path, 'val')
        raise_error_if_not_exists(self.rootpath)
        raise_error_if_not_exists(self.train_path)
        raise_error_if_not_exists(self.val_path)
        
        # resnet=224, inception=299, diy_yolo
        resize_h, resize_w, inputlayer_h, inputlayer_w = dataresize

        # init transforms
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.train_diy_transforms = transforms.Compose([
            transforms.RandomAffine(degrees=[-15, 15], translate=[0.1, 0.1], scale=[0.9, 1.1]),
            transforms.Resize(size=(resize_h, resize_w)),
            # transforms.RandomRotation(10), # 5, 10
            transforms.CenterCrop(size=(inputlayer_h, inputlayer_w)),
            transforms.RandomHorizontalFlip(),# bad for charactor
            transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
            transforms.ToTensor(),
            normalize])
        val_transforms = transforms.Compose([
            # transforms.Resize(size=(resize_h, resize_w)),
            # transforms.CenterCrop(size=(inputlayer_h, inputlayer_w)),
            transforms.Resize(size=(inputlayer_h, inputlayer_w)),
            transforms.ToTensor(),
            normalize])
    
        # train_diy_dataloader
        self.dbparser = diy_folder.DatasetFolderParsing(self.train_path)
        number_dict = {key : 0 for key in self.dbparser.class_to_idx.keys()}
        self.base_number_specifier = diy_folder.BaseNumberSpecifier(number_dict, base_number)
        self.train_diy_dataset = diy_folder.ImageFolder_SpecifiedNumber(
            self.dbparser,
            number_dict=self.base_number_specifier.class_to_number_dict,
            transform=self.train_diy_transforms)
        self.train_sampler = None # do not using distributed mode
        self.train_diy_loader = torch.utils.data.DataLoader(
            self.train_diy_dataset,
            batch_size=batch_size,
            shuffle=(self.train_sampler is None),
            num_workers=workers, pin_memory=True, sampler=self.train_sampler)
        
        # val_*_loaders
        self.val_train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(self.train_path, val_transforms),
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers, pin_memory=True)
        self.val_val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(self.val_path, val_transforms),
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers, pin_memory=True)
        
    def update_train_diy_loader(self, top1_dict):
        self.base_number_specifier.update(top1_dict)
        number_dict = self.base_number_specifier.class_to_number_dict
        
        # diy dataloader
        print("number_dict = {}".format(number_dict))
        self.train_diy_dataset = diy_folder.ImageFolder_SpecifiedNumber(
            self.dbparser,
            number_dict=self.base_number_specifier.class_to_number_dict,
            transform=self.train_diy_transforms)
        self.train_sampler = None # do not using distributed mode
        self.train_diy_loader = torch.utils.data.DataLoader(
            self.train_diy_dataset,
            batch_size=self.batch_size,
            shuffle=(self.train_sampler is None),
            num_workers=self.workers, pin_memory=True, sampler=self.train_sampler)


# max_base_number is the base_number of the label with the fewest classes, keep this val
# adaptive aims to keep trainloader batches balance.
class MTMCDataloader():
    def __init__(self, path, dataresize=[256,256,224,224], batch_size=32, workers=4, max_base_number=100):
        # directory_check
        self.rootpath = path
        raise_error_if_not_exists(self.rootpath)
        labels, label_to_idx = diy_folder.find_classes(self.rootpath)
        if len(labels) == 0:
            raise(RuntimeError("Dataset path = {} has no folder as task-label.".format(self.rootpath)))
        self.labels = labels
        self.label_to_idx = copy.deepcopy(label_to_idx)
        
        # get_mtmc_tree
        self.get_mtmc_tree()
        
        # N * slmcdataloader_dict
        self.slmcdataloader_dict = copy.deepcopy(label_to_idx)
        for label in self.slmcdataloader_dict.keys():
            print('------MTMCDataloader::create_SLMCDataloader({})'.format(label))
            self.slmcdataloader_dict[label] = SLMCDataloader(
                    os.path.join(self.rootpath, label),
                    dataresize = dataresize,
                    batch_size=batch_size,
                    workers=workers,
                    base_number=self.get_suitable_base_number(label, max_base_number))

    def get_mtmc_tree(self):
        self.mtmc_tree = copy.deepcopy(self.label_to_idx)
        self.min_class_number = sys.maxsize # no sys.maxint in py3
        for label in self.mtmc_tree.keys():
            classes, class_to_idx = diy_folder.find_classes(os.path.join(self.rootpath, label, 'train'))
            val_classes, val_class_to_idx = diy_folder.find_classes(os.path.join(self.rootpath, label, 'val'))
            if not class_to_idx == val_class_to_idx:
                print('train_class_to_idx = {}'.format(class_to_idx))
                print('val_class_to_idx = {}'.format(val_class_to_idx))
                raise(RuntimeError("train_class_to_idx != val_class_to_idx."))
            self.mtmc_tree[label] = copy.deepcopy(class_to_idx)
            if self.min_class_number > len(classes):
                self.min_class_number = len(classes)

    def get_suitable_base_number(self, label, max_base_number):
        class_number = len(self.mtmc_tree[label].keys())
        return (max_base_number * self.min_class_number) // class_number # py3
        
        
    def update_train_diy_loader(self, label_top1_dict):
        for label in label_top1_dict.keys():
            print('------MTMCDataloader::update_train_diy_loader({})'.format(label))
            self.slmcdataloader_dict[label].update_train_diy_loader(label_top1_dict[label])

def make_fake_label_top1_dict(label_top1_dict):
    for label in label_top1_dict.keys():
        for c in label_top1_dict[label].keys():
            label_top1_dict[label][c] = random.random()
            
            
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
            description='python main.py --path=data'
            )
    parser.add_argument(
                        "--path",
                        default='/Users/baiqi/data/pants',
                        type=str,
                        )
    args = parser.parse_args()
    print('args.path = {}'.format(args.path))
    path = args.path
    
    '''--------------------------------------'''

    '''SLMCDataloader--------------------------------------'''
    # SLMCDataloader
    slmcdataloader = SLMCDataloader(os.path.join(path, 'length'), batch_size=32, workers=4, base_number=128)
    '''
    for i, (input, target) in enumerate(slmcdataloader.train_diy_loader):
        print(target)
        print(i)
        '''
    print('sldataloader.train_diy_loader.__len__() = {}'.format(slmcdataloader.train_diy_loader.__len__()))
    print('sldataloader.val_train_loader.__len__() = {}'.format(slmcdataloader.val_train_loader.__len__()))
    print('sldataloader.val_val_loader.__len__() = {}'.format(slmcdataloader.val_val_loader.__len__()))


    '''MTMCDataloader--------------------------------------'''
    # MTMCDataloader
    mtmcdataloader = MTMCDataloader(path, batch_size=32, workers=4, max_base_number=100)
    for label in mtmcdataloader.label_to_idx.keys():
        print('---main::label={}'.format(label))
        slmcdataloader = mtmcdataloader.slmcdataloader_dict[label]
        print('sldataloader.train_diy_loader.__len__() = {}'.format(slmcdataloader.train_diy_loader.__len__()))
        print('sldataloader.val_train_loader.__len__() = {}'.format(slmcdataloader.val_train_loader.__len__()))
        print('sldataloader.val_val_loader.__len__() = {}'.format(slmcdataloader.val_val_loader.__len__()))
    
    # mtmcdataloader.update_train_diy_loader(label_top1_dict)
    for epoch in range(10):
        print('----------epoch = {}------------'.format(epoch))
        label_top1_dict = copy.deepcopy(mtmcdataloader.mtmc_tree)
        make_fake_label_top1_dict(label_top1_dict)
        mtmcdataloader.update_train_diy_loader(label_top1_dict)
        for label in mtmcdataloader.label_to_idx.keys():
            print('-----------main::label={}'.format(label))
            slmcdataloader = mtmcdataloader.slmcdataloader_dict[label]
            print('sldataloader.train_diy_loader.__len__() = {}'.format(slmcdataloader.train_diy_loader.__len__()))
            print('sldataloader.val_train_loader.__len__() = {}'.format(slmcdataloader.val_train_loader.__len__()))
            print('sldataloader.val_val_loader.__len__() = {}'.format(slmcdataloader.val_val_loader.__len__()))
            class_to_number_dict = slmcdataloader.base_number_specifier.class_to_number_dict
            print('class_to_number_dict = {}'.format(class_to_number_dict))


    