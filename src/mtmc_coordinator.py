#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 14:52:43 2018

Python 3.6.6 |Anaconda, Inc.| (default, Jun 28 2018, 11:07:29)

@author: pilgrim.bin@163.com
"""

import random

class SLMCCoordinator():
    def __init__(self, max_lens_list):
        # input params
        self.max_lens_list = max_lens_list
        self.iter_flag_list = []
        for idx in range(len(max_lens_list)):
           self.iter_flag_list += [idx] * max_lens_list[idx]
        random.shuffle(self.iter_flag_list)
        
if __name__ == '__main__':
    max_lens_list = [12, 20, 15]
    coordinator = SLMCCoordinator(max_lens_list)
    
    for i, iter_flag in enumerate(coordinator.iter_flag_list):
        print([i, iter_flag])
