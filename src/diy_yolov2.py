#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 14:52:43 2018

Python 3.6.6 |Anaconda, Inc.| (default, Jun 28 2018, 11:07:29)

@author: pilgrim.bin@163.com
"""

import torch.nn as nn

cfg = {
    'Yolov2_960x640':[[32, 3, 1], # <----960x640
                     'M',
                     [64, 3, 1],
                     'M',
                     [128, 3, 1],
                     [64,  1, 0],
                     [128, 3, 1],
                     'M',
                     [256, 3, 1],
                     [128, 1, 0],
                     [256, 3, 1],
                     'M',
                     [512, 3, 1],
                     [256, 1, 0],
                     [512, 3, 1],
                     [256, 1, 0],
                     [512, 3, 1],
                     'M',         # 5*M = <----30x20
                     [1024, 3, 0],
                     [512,  1, 0],
                     [1024, 3, 0],
                     'M',
                     [1024, 3, 0],
                     [512, 1, 0],
                     [1024, 3, 0]],
                      
    'Yolov2_768x512_raw':[[32, 3, 1], # <----768x512
                         'M',
                         [64, 3, 1],
                         'M',
                         [128, 3, 1],
                         [64,  1, 0],
                         [128, 3, 1],
                         'M',
                         [256, 3, 1],
                         [128, 1, 0],
                         [256, 3, 1],
                         'M',
                         [512, 3, 1],
                         [256, 1, 0],
                         [512, 3, 1],
                         'M',         # 5*M <----24x16
                         [1024, 3, 1],
                         [512,  1, 0],
                         [1024, 3, 1],
                         'M',         # 6*M <----12x8
                         [1024, 3, 1],
                         [512,  1, 0],
                         [1024, 3, 1],
                         'M',         # 7*M <----6x4
                         [1024, 3, 1],
                         [512,  1, 0],
                         [512, 3, 1],
                         'M'],         # 8*M <----3x2
                      
    'Yolov2_768x512_v2':[[32, 3, 1], # <----768x512
                         'M',
                         [64, 3, 1],
                         'M',
                         [128, 3, 1],
                         [64,  1, 0],
                         [128, 3, 1],
                         'M',
                         [128, 3, 1],
                         [64,  1, 0],
                         [128, 3, 1],
                         'M',
                         [256, 3, 1],
                         [128, 1, 0],
                         [256, 3, 1],
                         'M',         # 5*M <----24x16
                         [256, 3, 1],
                         [128, 1, 0],
                         [256, 3, 1],
                         'M',         # 6*M <----12x8
                         [512, 3, 1],
                         [256, 1, 0],
                         [512, 3, 1],
                         'M',         # 7*M <----6x4
                         [512, 3, 1],
                         [256,  1, 0],
                         [512, 3, 1],
                         'M'],         # 8*M <----3x2
                         
    # 'Yolov2_384x256_v3' --> batchsize=128, @Tesla P100-PCIE... cuMemory = 15743MiB / 16276MiB
    'Yolov2_384x256_v3':[[32, 3, 1], # <----384x256
                         'M',
                         [64, 3, 1],
                         'M',
                         [128, 3, 1],
                         [64,  1, 0],
                         [128, 3, 1],
                         'M',
                         [128, 3, 1],
                         [64,  1, 0],
                         [128, 3, 1],
                         'M',
                         [256, 3, 1],
                         [128, 1, 0],
                         [256, 3, 1],
                         'M',         # 5*M <----12x8
                         [256, 3, 1],
                         [128, 1, 0],
                         [256, 3, 1],
                         'M',         # 6*M <----6x4
                         [512, 3, 1],
                         [256, 1, 0],
                         [512, 3, 1],
                         'M']         # 7*M <----3x2
}

def make_yolov2_backbone_layers(cfg, batch_norm=True):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            out_channels, kernel_size, padding = v
            conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = out_channels
    return nn.Sequential(*layers)


class Yolov2Backbone(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(Yolov2Backbone, self).__init__()
        self.features = features
        self.classifier = nn.Linear(512 * 3 * 2, num_classes)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def diy_yolov2(pretrained=False, state_dict=None, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = Yolov2Backbone(make_yolov2_backbone_layers(cfg['Yolov2_384x256_v3']), **kwargs)
    if pretrained:
        model.load_state_dict(state_dict)
    return model






