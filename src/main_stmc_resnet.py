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

import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import folder_diy

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# data
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--base_number', default=10000, type=int,
                    help='base number of each class sample ')

# net
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--class_number', default=1000, type=int, metavar='N',
                    help='number of class (default: 1000)')

# training params
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
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

def main():
    global args, best_prec1
    args = parser.parse_args()

    args.distributed = args.world_size > 1
    
    print("args.distributed = {}".format(args.distributed))

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    class_number = args.class_number
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, class_number)

    if not args.distributed:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)

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
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    '''
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    '''
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(224), # raw = 256, CenterCrop is not a good trick.
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    val_trainset_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.Resize(224), # raw = 256, CenterCrop is not a good trick.
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    if args.evaluate:
        prec1, top1_dict = validate(val_loader, model, criterion, phase="VAL")
        return
    
    # diy dataloader
    base_number = args.base_number
    dbparser = folder_diy.DatasetFolderParsing(traindir)
    number_dict = {key : 0 for key in dbparser.class_to_idx.keys()}
    base_number_specifier = folder_diy.BaseNumberSpecifier(number_dict, base_number)
    train_diy_transforms = transforms.Compose([
        transforms.Resize(224), # raw = 256, CenterCrop is not a good trick.
        # transforms.RandomRotation(10),
        transforms.CenterCrop(224),
        # transforms.RandomHorizontalFlip(),# bad for charactor
        # transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
        transforms.ToTensor(),
        normalize])

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch)
        
        # diy dataloader
        print("number_dict = {}".format(number_dict))
        train_diy_dataset = folder_diy.ImageFolder_SpecifiedNumber(
            dbparser, number_dict=number_dict,
            transform=train_diy_transforms)
    
        train_diy_loader = torch.utils.data.DataLoader(
                train_diy_dataset,
                batch_size=args.batch_size, shuffle=(train_sampler is None),
                num_workers=args.workers, pin_memory=True, sampler=train_sampler)

        # train for one epoch
        #train(train_loader, model, criterion, optimizer, epoch)
        train(train_diy_loader, model, criterion, optimizer, epoch)
        
        # evaluate on trainset
        prec1, top1_dict = validate(val_trainset_loader, model, criterion, phase="TRAIN")
        # update sample number of each class
        if epoch % 20 == 0:
            base_number_specifier.update(top1_dict)
            number_dict=base_number_specifier.class_to_number_dict
        print("TRAIN top1_dict = {}".format(top1_dict))
        print("TRAIN number_dict = {}".format(number_dict))
        print("TRAIN top1_dict.values() = {}".format(top1_dict.values()))
        print("TRAIN number_dict.values() = {}".format(number_dict.values()))
        
        # evaluate on validation set
        prec1, top1_dict = validate(val_loader, model, criterion, phase="VAL")
        print("VAL top1_dict = {}".format(top1_dict))
        print("VAL top1_dict.values() = {}".format(top1_dict.values()))

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
            }, is_best,
            filename='checkpoint_{}.pth.tar'.format(args.arch))


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        [prec1, prec5], class_to = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))


def cout_confmatrix_tofile(ConfusionMatrix, classes, filename):
    cm = ConfusionMatrix
    print('------ConfusionMatrix[target[idx], class_to[idx]]:')
    print(cm)
    print('------ConfusionMatrix ratio per class:')
    cmf = cm.astype(np.float)
    ss = cmf.sum(1,keepdims=True)
    for idx in range(len(ss)):
        cmf[idx,] /= ss[idx]
    print(cmf)
    
    with open(filename, 'wb') as fp:
        # cm
        fp.write('ConfusionMatrix:\n')
        fp.write('classify_as:\t')
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
        fp.write('classify_as:\t')
        [fp.write('{}\t'.format(c)) for c in classes]
        fp.write('\n')
        for h in range(len(classes)):
            fp.write('{}\t'.format(classes[h]))
            for w in range(len(classes)):
                fp.write('{0:.3f}\t'.format(cmf[h,w]))
            fp.write('\n')
        fp.write('--------------------------------------')
        fp.write('\n')
        
        # Precision@1
        fp.write('Precision@1:\n')
        for h in range(len(classes)):
            fp.write('{}\t{}\n'.format(classes[h], cmf[h,h]))
        fp.write('--------------------------------------')
        fp.write('\n')
    return cmf
    


def validate(val_loader, model, criterion, phase="VAL"):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

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
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            [prec1, prec5], class_to = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test-{0}: [{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                              phase, i, len(val_loader),
                              batch_time=batch_time,
                              loss=losses,
                              top1=top1, top5=top5))
            
            # save result to ConfusionMatrix
            label = target.cpu().numpy()
            for idx in range(len(class_to)):
                ConfusionMatrix[label[idx], class_to[idx]] += 1

        print(' * {} Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(phase, top1=top1, top5=top5))
        
        filename = 'val_{}_ConfusionMatrix_'.format(phase) \
            + time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time())) \
            + '.txt'
        filename = os.path.join(vals_log_path, filename)
        cmf = cout_confmatrix_tofile(ConfusionMatrix, classes, filename)
        
    top1_dict = {classes[i] : cmf[i,i] for i in range(len(classes))}
    return (top1.avg, top1_dict)


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

