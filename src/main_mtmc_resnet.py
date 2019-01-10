#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 17:07:39 2018

Python 3.6.6 |Anaconda, Inc.| (default, Jun 28 2018, 11:07:29)

reverse based on pytorch::examples/imagenet
$  conda list | grep torch
pytorch                   0.4.1           py36_cuda0.0_cudnn0.0_1    pytorch
torchvision               0.2.1                    py36_1    pytorch

@author: pilgrim.bin@163.com
"""
import argparse
import os
import shutil
import time
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
#import torchvision.transforms as transforms
#import torchvision.datasets as datasets
import torchvision.models as models
import mtmcmodel as mtmcmodel
import mtmcconfig as mtmcconfig
import diy_yolov2

#import folder_diy
from mtmc_coordinator import SLMCCoordinator

tv_model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# data
parser.add_argument('--data', metavar='DIR',
                    default='/Users/baiqi/data/pants',
                    help='path to dataset')
parser.add_argument('--max_base_number', default=5000, type=int,
                    #help='base number of each class sample.')
                    help='max_base_number is the base_number of the label with the fewest classes.')

# net
'''
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=tv_model_names,
                    help='model architecture: ' +
                        ' | '.join(tv_model_names) +
                        ' (default: resnet18)')
'''
parser.add_argument('--arch', metavar='ARCH', default='yolov2',
                    # default='resnet18',
                    help='model architecture: tv.models or diy_model.')

'''
parser.add_argument('--class_number', default=1000, type=int, metavar='N',
                    help='number of class (default: 1000)')
'''
# add for diy models
parser.add_argument('--dataloader_resize_h', default=256, type=int)
parser.add_argument('--dataloader_resize_w', default=256, type=int)
parser.add_argument('--inputlayer_h', default=256, type=int)
parser.add_argument('--inputlayer_w', default=256, type=int)
parser.add_argument('--fc_features', default=512, type=int, help='net:input layer size and model framework defines the input-fc-features')


# training params
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--fc_epochs', default=50, type=int, metavar='N',
                    help='number of epochs to update optimizer.')

parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')

best_prec1 = 0

# vals log
vals_log_path = "vals_log"
if not os.path.exists(vals_log_path):
    os.mkdir(vals_log_path)
    
def cout_info_BaseNumberSpecifier(mtmcdataloader):
    for label in mtmcdataloader.labels:
        print('-------cout_info_BaseNumberSpecifier::{}--------'.format(label))
        print("TRAIN-Label = {} top1_dict = {}".format(label,
              mtmcdataloader.slmcdataloader_dict[label].base_number_specifier.class_to_number_dict))
        print("TRAIN-Label = {} number_dict = {}".format(label,
              mtmcdataloader.slmcdataloader_dict[label].base_number_specifier.class_to_prec_dict))
        

def main():
    global args, best_prec1
    args = parser.parse_args()

    print('args = {}'.format(args))

    args.distributed = args.world_size > 1
    print("args.distributed = {}".format(args.distributed))
    if args.distributed:
        raise(RuntimeError("Distributed Mode is Unsupported Currently."))
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)
    train_sampler = None

    '''-------------------------------------
        # create dataloader
    ----------------------------------------'''
    # resize params
    dataresize = [args.dataloader_resize_h, args.dataloader_resize_w, args.inputlayer_h, args.inputlayer_w]
    # MTMCDataloader
    mtmcdataloader = mtmcconfig.MTMCDataloader(
            args.data, # data root path
            dataresize=dataresize,
            batch_size=args.batch_size,
            workers=args.workers,
            max_base_number=args.max_base_number)
    print('INFO: = mtmcdataloader.mtmc_tree = {}'.format(mtmcdataloader.mtmc_tree))
    print('INFO: = mtmcdataloader.label_to_idx = {}'.format(mtmcdataloader.label_to_idx))
    
    label_list = mtmcdataloader.label_to_idx.keys()
    #label_list.sort() # py2
    label_list = sorted(label_list)
    class_numbers = []
    for label in label_list:
        class_numbers.append(len(mtmcdataloader.mtmc_tree[label]))

    '''-------------------------------------
        # create model
    ----------------------------------------'''
    if args.arch in tv_model_names:
        # using torchvision modles, resnet or inception
        if args.pretrained:
            print("=> using pre-trained model '{}'".format(args.arch))
            model = models.__dict__[args.arch](pretrained=True)
        else:
            print("=> creating model '{}'".format(args.arch))
            model = models.__dict__[args.arch]()
        # raw mc
        # fc_features = model.fc.in_features # it is 512 if using resnet18_224x224
        # model.fc = nn.Linear(fc_features, class_number)
        # new mtmc
        fc_features = args.fc_features
        model.fc = mtmcmodel.BuildMultiLabelModel(fc_features, class_numbers)
    elif 'yolo' in args.arch.lower(): # yolo_vx_h_w = yolov123_768x512
        g_inputlayer_heigth, g_inputlayer_width = args.arch.lower().split('_')[-1].split('x')
        g_inputlayer_heigth = int(g_inputlayer_heigth)
        g_inputlayer_width  = int(g_inputlayer_width)
        model = diy_yolov2.diy_yolov2(pretrained=False, num_classes=1000, init_weights=True)
        fc_features = args.fc_features
        model.classifier = mtmcmodel.BuildMultiLabelModel(fc_features, class_numbers)
    else:
        raise(RuntimeError("Unknown model arch = {}.".format(args.arch)))

    if not args.distributed:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)


    '''-------------------------------------
        # set criterion & optimizer
    ----------------------------------------'''
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            if args.start_epoch >= args.fc_epochs:
                optimizer = torch.optim.SGD(model.module.fc.parameters(), args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    '''
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize]))
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    '''

    if args.evaluate:
        prec1_avg, label_prec1_top1_dict_dict, label_top1_dict = mtmc_validate(mtmcdataloader, model, criterion, phase="VAL")
        print('label_prec1_top1_dict_dict = {}'.format(label_prec1_top1_dict_dict))
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch)
        
        '''-------------------------------------
        # update criterion & optimizer
        # 可以分三阶段：FC, Base + FC, FC.
        ----------------------------------------'''
        # define loss function (criterion) and optimizer
        if epoch == args.fc_epochs:
            # criterion = nn.CrossEntropyLoss().cuda()
            optimizer = torch.optim.SGD(model.module.fc.parameters(), args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)
        
        # mtmc_coordinator
        max_lens_list = []
        for label in label_list:
            dataloader = mtmcdataloader.slmcdataloader_dict[label]
            max_lens_list.append(dataloader.train_diy_loader.__len__()) 
        print("SLMCCoordinator::max_lens_list = {}".format(max_lens_list))
        coordinator = SLMCCoordinator(max_lens_list)

        # train for one epoch
        mtmctrain(mtmcdataloader, model, criterion, optimizer, epoch, coordinator)
        
        # evaluate on trainset
        ''' # no need to waste time testing trainset
        prec1_avg, label_prec1_top1_dict_dict, label_top1_dict = mtmc_validate(mtmcdataloader, model, criterion, phase="TRAIN")
        print('Train - label_prec1_top1_dict_dict = {}'.format(label_prec1_top1_dict_dict))
        '''
        
        # update sample number of each class
        if epoch > 0 and epoch % 20 == 0:
            prec1_avg, label_prec1_top1_dict_dict, label_top1_dict = mtmc_validate(mtmcdataloader, model, criterion, phase="TRAIN")
            print('Train - label_prec1_top1_dict_dict = {}'.format(label_prec1_top1_dict_dict))
            mtmcdataloader.update_train_diy_loader(label_top1_dict)
            
        # cout_info_BaseNumberSpecifier(mtmcdataloader)
        cout_info_BaseNumberSpecifier(mtmcdataloader)
        
        # evaluate on validation set
        prec1_avg, label_prec1_top1_dict_dict, label_top1_dict = mtmc_validate(mtmcdataloader, model, criterion, phase="VAL")
        print('VAL - label_prec1_top1_dict_dict = {}'.format(label_prec1_top1_dict_dict))

        # remember best prec@1 and save checkpoint
        is_best = prec1_avg > best_prec1
        best_prec1 = max(prec1_avg, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
            }, is_best,
            filename='checkpoint_{}.pth.tar'.format(args.arch))
        if epoch > 0 and epoch % 10 == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
                }, False,
                filename='checkpoint_{}_{}.pth.tar'.format(args.arch, epoch + 1))


def get_dict_key(dict, value):
    for k in dict.keys():
        if dict[k] == value:
            return k
    return None


# it goes wrong if using model = torch.nn.DataParallel(model).cuda()
def backbone_zero_grad(model):
    model.avgpool.zero_grad()
    model.bn1.zero_grad()
    model.conv1.zero_grad()
    model.layer1.zero_grad()
    model.layer2.zero_grad()
    model.layer3.zero_grad()
    model.layer4.zero_grad()
    model.relu.zero_grad()
    model.maxpool.zero_grad()

def mtmctrain(mtmcdataloader, model, criterion, optimizer, epoch, coordinator):
    # vals log
    val_measure_dict = copy.deepcopy(mtmcdataloader.label_to_idx)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    for key in val_measure_dict.keys():
        losses = AverageMeter() # 0
        top1 = AverageMeter()   # 1
        topk = AverageMeter()   # 2
        val_measure_dict[key] = [losses, top1, topk]
    

    # switch to train mode
    model.train()
    
    # enumerate_list of traindata loader
    enumerate_list = []
    for label in mtmcdataloader.labels:
        enumerate_list.append(enumerate(mtmcdataloader.slmcdataloader_dict[label].train_diy_loader))

    end = time.time()
    for i, flag in enumerate(coordinator.iter_flag_list):
        #i, (input, target) = enumerate_list[flag].next() # py2
        i, (input, target) = next(enumerate_list[flag])
        label = get_dict_key(mtmcdataloader.label_to_idx, flag)
        
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)

        # compute output
        output = model(input)[flag]
        loss = criterion(output, target)

        # measure accuracy and record loss
        [prec1, preck], class_to = accuracy(output, target, topk=(1, 2))
        val_measure_dict[label][0].update(loss.item(), input.size(0))
        val_measure_dict[label][1].update(prec1[0], input.size(0))
        val_measure_dict[label][2].update(preck[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: {} - [{}][{}/{}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@2 {topk.val:.3f} ({topk.avg:.3f})'.format(
                         label, epoch, i, len(mtmcdataloader.slmcdataloader_dict[label].train_diy_loader),
                         batch_time=batch_time, data_time=data_time,
                         loss=val_measure_dict[label][0],
                         top1=val_measure_dict[label][1],
                         topk=val_measure_dict[label][2]))


def cout_confmatrix_tofile(ConfusionMatrix, classes, filename):
    cm = ConfusionMatrix
    print('------ConfusionMatrix[target[idx], class_to[idx]]:')
    print(cm)
    print('------Recall = ConfusionMatrix ratio per class:')
    cmf = cm.astype(np.float)
    ss = cmf.sum(1,keepdims=True)
    for idx in range(len(ss)):
        cmf[idx,] /= ss[idx]
    print(cmf)
    
    # Precision
    print('------Precision matrix @ ConfusionMatrix :')
    cmf_prec = cm.astype(np.float)
    ss = cmf_prec.sum(0,keepdims=True)
    precs_top1 = []
    for idx in range(len(ss[0])):
        cmf_prec[:,idx] /= ss[0][idx]
        precs_top1.append(cmf_prec[idx,idx])
    print('precs_top1 = {}'.format(precs_top1))
    
    with open(filename, 'w') as fp: # using 'wb' in py2
        # cm
        fp.write('ConfusionMatrix:\n')
        fp.write('gt\\pred:\t')
        [fp.write('{}\t'.format(c)) for c in classes]
        fp.write('\n')
        for h in range(len(classes)):
            fp.write('{}\t'.format(classes[h]))
            for w in range(len(classes)):
                fp.write('{}\t'.format(cm[h,w]))
            fp.write('\n')
        fp.write('--------------------------------------')
        fp.write('\n')
        
        # cmf
        fp.write('ConfusionMatrix classify ratio:\n')
        fp.write('gt\\pred:\t')
        [fp.write('{}\t'.format(c)) for c in classes]
        fp.write('\n')
        for h in range(len(classes)):
            fp.write('{}\t'.format(classes[h]))
            for w in range(len(classes)):
                fp.write('{0:.3f}\t'.format(cmf[h,w]))
            fp.write('\n')
        fp.write('--------------------------------------')
        fp.write('\n')
        
        # Recall
        fp.write('Recall:\n')
        for h in range(len(classes)):
            fp.write('{}\t{}\n'.format(classes[h], cmf[h,h]))
        fp.write('--------------------------------------')
        fp.write('\n')
        
        # Precision@1
        fp.write('Precision@1:\n')
        for h in range(len(classes)):
            fp.write('{}\t{}\n'.format(classes[h], cmf_prec[h,h]))
        fp.write('--------------------------------------')
        fp.write('\n')
    return precs_top1
    


def slmc_validate(val_loader, model, model_idx, criterion, phase="VAL"):
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    topk = AverageMeter()

    # switch to evaluate mode
    model.eval()
    
    classes = val_loader.dataset.classes
    class_number = len(classes)
    ConfusionMatrix = np.zeros((class_number, class_number), dtype=int)
    
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)[model_idx]
            loss = criterion(output, target)

            # measure accuracy and record loss
            [prec1, preck], class_to = accuracy(output, target, topk=(1, 2))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            topk.update(preck[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test-{0}: [{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@2 {topk.val:.3f} ({topk.avg:.3f})'.format(
                              phase, i, len(val_loader),
                              batch_time=batch_time,
                              loss=losses,
                              top1=top1, topk=topk))
            
            # save result to ConfusionMatrix
            label = target.cpu().numpy()
            for idx in range(len(class_to)):
                ConfusionMatrix[label[idx], class_to[idx]] += 1

        print(' * {} Prec@1 {top1.avg:.3f} Prec@2 {topk.avg:.3f}'
              .format(phase, top1=top1, topk=topk))
        
        filename = 'val_{}_ConfusionMatrix_'.format(phase) \
            + time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time())) \
            + '.txt'
        filename = os.path.join(vals_log_path, filename)
        precs_top1 = cout_confmatrix_tofile(ConfusionMatrix, classes, filename)
        
    top1_dict = {classes[i] : precs_top1[i] for i in range(len(classes))}
    return (top1.avg, top1_dict)


def mtmc_validate(mtmcdataloader, model, criterion, phase="VAL"):
    label_prec1_top1_dict_dict = copy.deepcopy(mtmcdataloader.label_to_idx)
    label_top1_dict = copy.deepcopy(mtmcdataloader.label_to_idx)
    prec1_avg = 0
    for label in mtmcdataloader.label_to_idx.keys():
        if phase == "VAL":
            val_loader = mtmcdataloader.slmcdataloader_dict[label].val_val_loader
        else: # 'TRAIN'
            val_loader = mtmcdataloader.slmcdataloader_dict[label].val_train_loader
        model_idx = mtmcdataloader.label_to_idx[label]
        prec1, top1_dict = slmc_validate(val_loader, model, model_idx, criterion, phase='_'.join([phase, label]))
        prec1_avg += prec1
        label_prec1_top1_dict_dict[label] = tuple([prec1, top1_dict])
        label_top1_dict[label] = top1_dict
    return (prec1_avg / float(len(label_prec1_top1_dict_dict.keys())), label_prec1_top1_dict_dict, label_top1_dict)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best_' + filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 40))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        class_to = pred[0].cpu().numpy()

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res, class_to


if __name__ == '__main__':
    main()

