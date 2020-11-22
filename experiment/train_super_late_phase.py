# Adapted from https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
# Adapted from https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py

import argparse
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from utils import ensemble_util

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default='./datasets',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-bs', '--batch_split', default=2, type=int,
                    metavar='N', help='split factor of the batch when using small GPUs')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')

parser.add_argument('--spec_lr', default=0.001, type=float,
                    help='initial learning rate')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=1000, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--std', default=0, type=float,
                    help='std of noise')
parser.add_argument('--num_specialist', default=10, type=int,
                    help='number of specialist')
parser.add_argument('--average', default=True, type=bool,
                    help='average nested sgd')
parser.add_argument('--spawn_head', default=False, type=bool,
                    help='average nested sgd')
parser.add_argument('--steps', default='10', type=str,
                    help='average nested sgd')
parser.add_argument('--spec_steps', default='', type=str,
                    help='average nested sgd')
parser.add_argument('--anneal', default=False, type=bool,
                    help='average nested sgd')
parser.add_argument('--per_spec_optim', default=False, type=bool,
                    help='average nested sgd')

parser.add_argument('--id', default='', type=str,
                    help='average nested sgd')

best_prec1 = 0


def override_bn(model):
    return

def main():
    global args, best_prec1
    args = parser.parse_args()
    print(model_names)
    # Load pretrained model
    print("=> using pre-trained model '{}'".format(args.arch))
    model = models.__dict__[args.arch](pretrained=True)

    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageNet(root=args.data, split='train', download=None, transform=transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size // args.batch_split, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageNet(root=args.data, split='val', download=None, transform=transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=32, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    for n, m in model.named_modules():
        if n == 'module.fc':
            print("Found head")

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()

    lr = args.lr / args.batch_split / args.num_specialist if args.average else args.lr / args.batch_split
    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    specialist_modules = []
    found_head = False
    for n, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm.BatchNorm2d):
            specialist_modules.append(module)
        if isinstance(module, nn.modules.batchnorm.BatchNorm1d):
            specialist_modules.append(module)
        if n == 'module.fc' and args.spawn_head:
            specialist_modules.append(module)
            found_head = True

    if args.spawn_head:
        assert found_head

    assert len(specialist_modules) > 1
    print("Found {} convertible units".format(len(specialist_modules)))

    specialist_param = []
    per_specialist_param = [[] for _ in range(args.num_specialist)]
    for m in specialist_modules:
        ensemble_util.convert_specialist(m, args.num_specialist, args.std)
        for s in range(args.num_specialist):
            for p in m.specialist_modules[s].parameters():
                per_specialist_param[s].append(p)

    for s in range(args.num_specialist):
        specialist_param += per_specialist_param[s]

    spec_lr = args.spec_lr / args.batch_split

    specialist_optimizer = []
    if not args.per_spec_optim:
        specialist_optimizer = torch.optim.SGD(specialist_param, spec_lr,
                                               momentum=args.momentum,
                                               weight_decay=args.weight_decay)
    else:
        for s in range(args.num_specialist):
            specialist_optimizer.append(torch.optim.SGD(per_specialist_param[s], spec_lr,
                                                        momentum=args.momentum,
                                                        weight_decay=args.weight_decay))

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    filename = 'arch_' + str(args.arch)
    filename += '_spec_' + str(args.num_specialist)
    filename += '_std_' + str(args.std)
    filename += '_lr_' + str(args.lr)
    filename += '_speclr_' + str(args.spec_lr)
    filename += '_step_' + str(args.steps)
    filename += '_specstep_' + str(args.spec_steps)
    filename += '_head_' + str(args.spawn_head)
    filename += '_avr_' + str(args.average)
    filename += '_anneal_' + str(args.anneal)
    filename += '_specoptim_' + str(args.per_spec_optim)
    filename += '_' + str(args.id)

    log_file_name = filename + '_performance.txt'
    filename += '_checkpoint.pth.tar'
    print(filename)

    log_file = open(log_file_name, 'w')

    steps = args.steps.split(",") if args.steps != "" else []
    specialist_steps = args.spec_steps.split(",") if args.spec_steps != "" else []
    print(steps)
    print(specialist_steps)

    for epoch in range(args.start_epoch, args.epochs):
        print("Adjust meta LR")
        adjust_learning_rate(lr, optimizer, epoch, steps, args.anneal)

        print("Adjust specialist LR")
        if not args.per_spec_optim:
            adjust_learning_rate(spec_lr, specialist_optimizer, epoch, specialist_steps, args.anneal)
        else:
            for o in specialist_optimizer:
                adjust_learning_rate(spec_lr, o, epoch, specialist_steps, args.anneal)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, specialist_optimizer, specialist_modules, args.batch_split,
              args.per_spec_optim, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)
        print('Epoch:\t{}\tPrecision\t{}'.format(epoch, prec1), file=log_file)
        log_file.flush()

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=filename)


def check_grad(specialist_modules):
    if True:
        foobar = specialist_modules[0]
        for i, s in enumerate(foobar.specialist_modules):
            grad_norm = 0
            has_grad = False
            for p in s.parameters():
                if p.grad is not None:
                    grad_norm += (p.grad * p.grad).sum().item()
                    has_grad = True
            if not has_grad:
                print("Specialist {} has no grad".format(i))
            else:
                print("Specialist {} has grad {}".format(i, grad_norm))
        grad_norm = 0
        has_grad = False
        for p in foobar.parameters():
            if p.grad is not None:
                grad_norm += (p.grad * p.grad).sum().item()
                has_grad = True
        if not has_grad:
            print("Overall module has no grad")
        else:
            print("Overall module has grad {}".format(grad_norm))


def train(train_loader, model, criterion, optimizer, bn_optimizers, batchnorm_units, accumulate_step, per_spec_optim,
          epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    curr_specialist = 0
    num_specialist = batchnorm_units[0].num_specialist

    for i, (input, target) in enumerate(train_loader):
        if per_spec_optim:
            bn_optimizer = bn_optimizers[curr_specialist]
        else:
            bn_optimizer = bn_optimizers
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        if i % accumulate_step == 0:
            bn_optimizer.zero_grad()
            for bn in batchnorm_units:
                bn.curr_specialist = curr_specialist
            if curr_specialist == 0:
                optimizer.zero_grad()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        loss.backward()

        if (i + 1) % accumulate_step == 0:
            #            check_grad(batchnorm_units)
            bn_optimizer.step()
            if curr_specialist + 1 == num_specialist:
                optimizer.step()
            curr_specialist = (curr_specialist + 1) % num_specialist
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


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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


def adjust_learning_rate(lr, optimizer, epoch, steps, anneal):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    idx = 0
    for s in steps:
        if int(s) <= epoch:
            idx += 1
            lr *= 0.1
    if anneal and len(steps) > idx:
        start_epoch = 0 if idx == 0 else steps[idx - 1]
        lr *= (1 - 0.9 * (epoch - start_epoch) / (int(steps[idx]) - start_epoch))
    print("New LR:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
