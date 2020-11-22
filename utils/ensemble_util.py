from copy import deepcopy
import torch.nn as nn
import torch
import types
import itertools

class SpawnWrapper(nn.Module):
    def __init__(self, model):
        super(SpawnWrapper, self).__init__()
        self.model = model
        self.spawned = False

    def _collect_spawn_modules(self, spawn_head=False):
        self.spawn_modules = []
        head = None
        for n, module in self.model.named_modules():
            if isinstance(module, nn.modules.batchnorm.BatchNorm2d):
                self.spawn_modules.append(module)
            if isinstance(module, nn.modules.batchnorm.BatchNorm1d):
                self.spawn_modules.append(module)
            if isinstance(module, nn.modules.Linear):
                print(n)
                if n == 'module.classifier' or n == 'module.fc':
                    head = module
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
        self.spawned = True
        self.num_specialist = num_specialist

    def get_specialist_param(self):
        assert self.spawned
        return self.specialist_param

    def forward(self, x, specialist=None):
        assert self.num_specialist > specialist
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
            output = 0
            for i in range(self.num_specialist):
                output += self.specialist_modules[i](input) / self.num_specialist
            return output

    if isinstance(module, nn.modules.batchnorm.BatchNorm2d) or isinstance(module, nn.modules.batchnorm.BatchNorm1d):
        assert module.track_running_stats
    else:
        assert isinstance(module, nn.modules.Linear)

    with torch.no_grad():
        specialist_modules = []
        for i in range(n):
            next_bn = deepcopy(module)
            next_bn.weight.data += _perturbation(next_bn.weight, noise_std)
            next_bn.bias.data += _perturbation(next_bn.bias, noise_std)
            specialist_modules.append(next_bn)
        module.specialist_modules = nn.ModuleList(specialist_modules)

    module.forward = types.MethodType(new_forward, module)
    module.num_specialist = n
    module.curr_specialist = None
