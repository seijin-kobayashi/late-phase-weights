import torch
import numpy as np
import itertools
from nets import hyper_wrn

class OptimizerSet():
    def __init__(self, shared_p_optimizers=None, specific_p_optimizers=None,
                 legacy_mode=False):
        self.shared_p_optimizers = shared_p_optimizers
        self.specific_p_optimizers = specific_p_optimizers
        self.legacy_mode = legacy_mode

    def get_all_optimizers(self):
        optimizers = []
        if self.legacy_mode:
            optimizers += [self.shared_p_optimizers['other']]
            if self.specific_p_optimizers['coef']:
                optimizers += [self.specific_p_optimizers['coef']]
        else:
            if self.shared_p_optimizers['other']:
                optimizers += [self.shared_p_optimizers['other']]

            if self.shared_p_optimizers['coef']:
                optimizers += [self.shared_p_optimizers['coef']]

            if self.specific_p_optimizers['other']:
                for optim in self.specific_p_optimizers['other']:
                    optimizers += [optim]

            if self.specific_p_optimizers['coef']:
                for optim in self.specific_p_optimizers['coef']:
                    optimizers += [optim]

        return optimizers

    def get_base_lr_str(self):
        if self.legacy_mode:
            lr_str = str(self.shared_p_optimizers['other'].param_groups[0]['lr'])

        else:
            lr_str = ''
            if self.shared_p_optimizers['other']:
                lr_str += '[SHARED]=' + \
                    str(self.shared_p_optimizers['other'].param_groups[0]['lr']) + \
                    ' '
            
            if self.specific_p_optimizers['other']:
                for s, optim in enumerate(self.specific_p_optimizers['other']):
                    lr_str += '[SPECIALIST {}]='.format(s) + \
                        str(optim.param_groups[0]['lr']) + ' '
        
        return lr_str

    def get_coef_lr_str(self):
        if self.legacy_mode:
            lr_str = str(self.specific_p_optimizers['coef'].param_groups[0]['lr']) 

        else:
            lr_str = ''
            if self.shared_p_optimizers['coef']:
                lr_str += '[SHARED]=' + \
                    str(self.shared_p_optimizers['coef'].param_groups[0]['lr']) + \
                    ' '
            
            if self.specific_p_optimizers['coef']:
                for s, optim in enumerate(self.specific_p_optimizers['coef']):
                    lr_str += '[SPECIALIST {}]='.format(s) + \
                        str(optim.param_groups[0]['lr']) + ' '
        
        return lr_str

    def get_specific_coef_lr(self, specialist_idx=0):
        if self.legacy_mode:
            optim = self.specific_p_optimizers['coef']
        else:
            if self.specific_p_optimizers['coef']:
                optim = self.specific_p_optimizers['coef'][specialist_idx]
            else:
                raise Exception("Model has no specific coefficients.")
        
        return optim.param_groups[0]['lr']

    def get_specific_coef(self, specialist_idx=None):
        if self.legacy_mode:
            optim = self.specific_p_optimizers['coef']
            return optim.param_groups[0]['params']
        else:
            if self.specific_p_optimizers['coef']:
                if specialist_idx:
                    optim = self.specific_p_optimizers['coef'][specialist_idx]
                    return optim.param_groups[0]['params']
                else:
                    param_list = []
                    for s in range(len(self.specific_p_optimizers['coef'])):
                        optim = self.specific_p_optimizers['coef'][s]
                        param_list += optim.param_groups[0]['params']
                    return param_list
            else:
                return []

    def get_shared(self):
        if self.legacy_mode:
            optim = self.shared_p_optimizers['other']
            return optim.param_groups[0]['params']
        else:
            param_list = []
            if self.shared_p_optimizers['other']:
                optim = self.shared_p_optimizers['other']
                param_list += optim.param_groups[0]['params']

            if self.shared_p_optimizers['coef']:
                optim = self.shared_p_optimizers['other']
                param_list += optim.param_groups[0]['params']
                
            return param_list

    def zero_grad(self):
        if self.legacy_mode:
            self.shared_p_optimizers['other'].zero_grad()
            if self.specific_p_optimizers['coef']:
                self.specific_p_optimizers['coef'].zero_grad()
        else:
            if self.shared_p_optimizers['other']:
                self.shared_p_optimizers['other'].zero_grad()
            
            if self.shared_p_optimizers['coef']:
                self.shared_p_optimizers['coef'].zero_grad()

            if self.specific_p_optimizers['other']:
                for optim in self.specific_p_optimizers['other']:
                    optim.zero_grad()

            if self.specific_p_optimizers['coef']:
                for optim in self.specific_p_optimizers['coef']:
                    optim.zero_grad()

    def step(self):
        if self.legacy_mode:
            self.shared_p_optimizers['other'].step()
            if self.specific_p_optimizers['coef']:
                self.specific_p_optimizers['coef'].step()
        else:
            if self.shared_p_optimizers['other']:
                self.shared_p_optimizers['other'].step()
            
            if self.shared_p_optimizers['coef']:
                self.shared_p_optimizers['coef'].step()

            if self.specific_p_optimizers['other']:
                for optim in self.specific_p_optimizers['other']:
                    optim.step()

            if self.specific_p_optimizers['coef']:
                for optim in self.specific_p_optimizers['coef']:
                    optim.step()

    def base_p_step(self):
        if self.legacy_mode:
            self.shared_p_optimizers['other'].step()
        else:
            if self.shared_p_optimizers['other']:
                self.shared_p_optimizers['other'].step()

            if self.specific_p_optimizers['other']:
                for optim in self.specific_p_optimizers['other']:
                    optim.step()

    def base_p_zero_grad(self):
        if self.legacy_mode:
            self.shared_p_optimizers['other'].zero_grad()
        else:
            if self.shared_p_optimizers['other']:
                self.shared_p_optimizers['other'].zero_grad()

            if self.specific_p_optimizers['other']:
                for optim in self.specific_p_optimizers['other']:
                    optim.zero_grad()

    def coef_p_step(self, specialist_idx=None):
        if self.legacy_mode:
            self.specific_p_optimizers['coef'].step()
        else:
            if specialist_idx:
                self.specific_p_optimizers['coef'][specialist_idx].step()
            else:
                if self.shared_p_optimizers['coef']:
                    self.shared_p_optimizers['coef'].step()
            
                if self.specific_p_optimizers['coef']:
                    for optim in self.specific_p_optimizers['coef']:
                        optim.step()

    def coef_p_zero_grad(self, specialist_idx=None):
        if self.legacy_mode:
            self.specific_p_optimizers['coef'].zero_grad()
        else:
            if specialist_idx:
                self.specific_p_optimizers['coef'][specialist_idx].zero_grad()
            else:
                if self.shared_p_optimizers['coef']:
                    self.shared_p_optimizers['coef'].zero_grad()
            
                if self.specific_p_optimizers['coef']:
                    for optim in self.specific_p_optimizers['coef']:
                        optim.zero_grad()

    def shared_p_step(self):
        if self.legacy_mode:
            self.shared_p_optimizers['other'].step()
        else:
            if self.shared_p_optimizers['other']:
                self.shared_p_optimizers['other'].step()
            
            if self.shared_p_optimizers['coef']:
                self.shared_p_optimizers['coef'].step()

    def shared_p_zero_grad(self):
        if self.legacy_mode:
            self.shared_p_optimizers['other'].zero_grad()
        else:
            if self.shared_p_optimizers['other']:
                self.shared_p_optimizers['other'].zero_grad()
            
            if self.shared_p_optimizers['coef']:
                self.shared_p_optimizers['coef'].zero_grad()

    def specific_p_step(self, specialist_idx=None):
        if self.legacy_mode:
            self.specific_p_optimizers['coef'].step()
        else:
            if specialist_idx:
                if self.specific_p_optimizers['coef']:
                    self.specific_p_optimizers['coef'][specialist_idx].step()
                if self.specific_p_optimizers['other']:
                    self.specific_p_optimizers['other'][specialist_idx].step()
            else:
                if self.specific_p_optimizers['coef']:
                    for optim in self.specific_p_optimizers['coef']:
                        optim.step()

                if self.specific_p_optimizers['other']:
                    for optim in self.specific_p_optimizers['other']:
                        optim.step()

    def specific_p_zero_grad(self, specialist_idx=None):
        if self.legacy_mode:
            self.specific_p_optimizers['coef'].zero_grad()
        else:
            if specialist_idx:
                if self.specific_p_optimizers['coef']:
                    self.specific_p_optimizers['coef'][specialist_idx].zero_grad()
                if self.specific_p_optimizers['other']:
                    self.specific_p_optimizers['other'][specialist_idx].zero_grad()
            else:
                if self.specific_p_optimizers['coef']:
                    for optim in self.specific_p_optimizers['coef']:
                        optim.zero_grad()

                if self.specific_p_optimizers['other']:
                    for optim in self.specific_p_optimizers['other']:
                        optim.zero_grad()


def create_multistep_scheduler(optimizer, params, milestones, gamma):
    m = np.array(milestones)
    lambda_epoch = lambda e: gamma ** np.sum(m <= e)

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda_epoch)

def create_annealing_scheduler(optimizer, params, lr,
                               annealing_start, annealing_end,
                               annealing_end_lr):
    def swa_schedule(epoch):
        lr_ratio = annealing_end_lr / lr
        if epoch <= annealing_start:
            factor = 1.0
        elif epoch <= annealing_end:
            factor = 1.0 - (1.0 - lr_ratio) * \
                    (epoch - annealing_start) / (annealing_end-annealing_start)
        else:
            factor = lr_ratio
        
        return factor

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=swa_schedule)

def create_scheduler(config, optimizer, params, coef=False):
    if coef:
        string_prefix = 'coef_'
    else:
        string_prefix  = ''

    if config['lr_scheduler'] == 'multistep':
        scheduler = \
            create_multistep_scheduler(
                optimizer,
                params,
                config[string_prefix + 'lr_multistep_milestones'],
                config[string_prefix + 'lr_multistep_gamma'])

    elif config['lr_scheduler'] == 'annealing':
        scheduler = \
            create_annealing_scheduler(
                optimizer,
                params,
                config[string_prefix + 'lr'],
                config[string_prefix + 'annealing_start'],
                config[string_prefix + 'annealing_end'],
                config[string_prefix + 'annealing_end_lr'])
    else:
        raise Exception('Unknown scheduler.')
    
    return scheduler

def create_SGD_optimizer(params, lr, weight_decay, momentum, nesterov):
    return torch.optim.SGD(params,
                           lr=lr,
                           momentum=momentum,
                           weight_decay=weight_decay,
                           nesterov=nesterov)

def create_Adam_optimizer(params, lr, weight_decay):
    return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

def create_optimizer(config, params, train_loader=None, coef=False, specialist=False):
    if coef:
        string_prefix = 'coef_'
    else:
        string_prefix  = ''
    if specialist:
        string_prefix = 'specialist_'

    if config['optimizer'] == 'SGD' or config['optimizer'] == 'SWA':
        optimizer = \
            create_SGD_optimizer(params,
                                 config[string_prefix + 'lr'],
                                 config[string_prefix + 'weight_decay'],
                                 config['momentum'],
                                 config['nesterov'])

    elif config['optimizer'] == 'Adam':
        optimizer = \
            create_Adam_optimizer(params,
                                  config[string_prefix + 'lr'],
                                  config[string_prefix + 'weight_decay'])

    else:
        raise Exception('Unknown optimizer.')
    
    return optimizer


# Code below is adapted from (Izmailov et al. 2018)
# https://github.com/timgaripov/swa
### SWA-related utilities

def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= 1.0 - alpha
        param1.data += param2.data * alpha

def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model, verbose=False, subset=None, **kwargs):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.
        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    num_batches = len(loader)

    with torch.no_grad():
        if subset is not None:
            num_batches = int(num_batches * subset)
            loader = itertools.islice(loader, num_batches)
        if verbose:

            loader = tqdm.tqdm(loader, total=num_batches)
        for input, _ in loader:
            input = input.cuda(non_blocking=True)
            input_var = torch.autograd.Variable(input)
            b = input_var.data.size(0)

            momentum = b / (n + b)
            for module in momenta.keys():
                module.momentum = momentum

            if isinstance(model, hyper_wrn.HyperWRN):
                for s in range(model.num_specialists):
                    model(input_var, s, **kwargs)
            else:
                model(input_var, **kwargs)
            n += b

    model.apply(lambda module: _set_momenta(module, momenta))