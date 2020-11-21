
from copy import deepcopy
import torch.nn as nn
import torch
import types
import itertools

class SpawnWrapper(nn.Module):
    def __init__(self, model):
        super(SpawnWrapper, self).__init__()
        self.model=model
        self.spawned=False

    def _collect_spawn_modules(self, spawn_head=False):
        self.spawn_modules = []
        head=None
        for n, module in self.model.named_modules():
            if isinstance(module, nn.modules.batchnorm.BatchNorm2d):
                self.spawn_modules.append(module)
            if isinstance(module, nn.modules.batchnorm.BatchNorm1d):
                self.spawn_modules.append(module)
            if isinstance(module, nn.modules.Linear):
                print(n)
                if n=='module.classifier' or n=='module.fc':
                    head=module
                    print("Head found")
                    
        if spawn_head:
            assert (head is not None)
            self.spawn_modules.append(head)
        print("Found {} convertible units".format(len(self.spawn_modules)))

    def spawn_ensemble(self, num_specialist, std=0, spawn_head=False):
        print("Spawning ensemble")
        self._collect_spawn_modules(spawn_head=spawn_head)
        self.specialist_param = [[] for _ in range(num_specialist)]
        for m in self.spawn_modules:
            convert_specialist(m, num_specialist, std)
            for s in range(num_specialist):
                for p in m.specialist_modules[s].parameters():
                    self.specialist_param[s].append(p)
        self.spawned=True
        self.num_specialist=num_specialist

    def get_specialist_param(self):
        assert self.spawned
        return self.specialist_param

    def forward(self, x, specialist=None):
        assert self.num_specialist>specialist
        if self.spawned:
            for m in self.spawn_modules:
                m.curr_specialist = specialist
        return self.model(x)


def convert_specialist(module, n, noise_std):
    def _perturbation(source_tensor, noise_std):
        adjusted_noise_std = source_tensor.pow(2).mean().sqrt() * noise_std
        return torch.randn_like(source_tensor) * adjusted_noise_std

    def new_forward(self, input):
        if self.training:
            return self.specialist_modules[self.curr_specialist](input)
        else:
            output=0
            for i in range(self.num_specialist):
                output+=self.specialist_modules[i](input)/self.num_specialist
            return output

    if isinstance(module, nn.modules.batchnorm.BatchNorm2d) or isinstance(module, nn.modules.batchnorm.BatchNorm1d):
        assert module.track_running_stats
    else:
        assert isinstance(module, nn.modules.Linear)

    with torch.no_grad():
        specialist_modules=[]
        for i in range(n):
            next_bn=deepcopy(module)
            next_bn.weight.data+=_perturbation(next_bn.weight, noise_std)
            next_bn.bias.data+=_perturbation(next_bn.bias, noise_std)
            specialist_modules.append(next_bn)
        module.specialist_modules=nn.ModuleList(specialist_modules)

    module.forward = types.MethodType(new_forward, module)
    module.num_specialist=n
    module.curr_specialist=None


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


def bn_update(loader, model, subset=None, **kwargs):
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
        for input, _ in loader:
            input = input.cuda(non_blocking=True)
            input_var = torch.autograd.Variable(input)
            b = input_var.data.size(0)

            momentum = b / (n + b)
            for module in momenta.keys():
                module.momentum = momentum

            if model.spawned:
                for s in range(model.num_specialist):
                    model(input_var, s, **kwargs)
            else:
                model(input_var, **kwargs)
            n += b

    model.apply(lambda module: _set_momenta(module, momenta))




