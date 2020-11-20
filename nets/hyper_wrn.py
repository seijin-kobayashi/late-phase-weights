#!/usr/bin/env python3
# Please do not redistribute.

import math
from warnings import warn
from collections import defaultdict
from termcolor import cprint
from nets.hyper_layers import *

# Class inspired by (Savarese & Maire, ICLR 2019)
# https://github.com/lolemacs/soft-sharing
class Block(nn.Module):
    def __init__(self, in_planes, out_planes, stride,
                 num_specialists,
                 bank_registry=None,
                 specialist_batchnorm=False,
                 hard_sharing=False,
                 use_template_bank=False,
                 num_templates=None):
        super(Block, self).__init__()
        template_split_factor=2
        self.equalInOut = (in_planes == out_planes)
        self.bank_registry = bank_registry

        self.bn1 = SBatchNorm(
            in_planes,
            num_specialists=num_specialists,
            hard_sharing=not specialist_batchnorm)

        self.conv1 = SConv2d(in_planes, out_planes, stride=stride, bank_registry=self.bank_registry,
                             num_specialists=num_specialists,
                             hard_sharing=hard_sharing,
                             use_template_bank=use_template_bank,
                             template_split_factor=math.ceil(template_split_factor/(out_planes//in_planes)),
                             num_templates=num_templates if out_planes//in_planes>2 else num_templates*template_split_factor)

        self.bn2 = SBatchNorm(
            out_planes,
            num_specialists=num_specialists,
            hard_sharing=not specialist_batchnorm)

        self.conv2 = SConv2d(out_planes, out_planes, bank_registry=self.bank_registry,
                             num_specialists=num_specialists,
                             hard_sharing=hard_sharing,
                             use_template_bank=use_template_bank,
                             template_split_factor=template_split_factor,
                             num_templates=num_templates*template_split_factor)

        self.relu = nn.ReLU(inplace=True)

        self.convShortcut = SConv2d(in_planes, out_planes, kernel_size=1,
                    stride=stride, padding=0, bank_registry=self.bank_registry,
                    num_specialists=num_specialists,
                    hard_sharing=hard_sharing,
                    use_template_bank=use_template_bank,
                    num_templates=num_templates) if not self.equalInOut else None

    def forward(self, x, specialist, **kwargs):
        residual = x
        out = self.relu(self.bn1(x, specialist, **kwargs))
        if not self.equalInOut:
            residual = out
        conv1_out = self.conv1(out, specialist, **kwargs)

        out = self.conv2(self.relu(self.bn2(conv1_out, specialist, **kwargs)),specialist, **kwargs)

        if self.convShortcut is not None:
            residual = self.convShortcut(residual, specialist, **kwargs)
        return out + residual

class MultiInputSequential(nn.Sequential):
    def forward(self, x, specialist, **kwargs):
        for module in self._modules.values():
            x = module(x, specialist, **kwargs)
        return x

class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.shape[0], *self.shape)

class HyperWRN(nn.Module):
    def __init__(self,
                 depth, width, num_classes,
                 in_shape=[3, 32, 32],
                 num_specialists=1,
                 compression_factor=1,
                 hard_sharing=[
                     "input_layer",
                     "adapter",
                     "conv_block",
                     "head",
                     "batchnorm"],
                 use_template_bank=[
                     "input_layer", "adapter", "conv_block", "head"],
                 start_tied=False,
                 noise_std_init=0):
        super(HyperWRN, self).__init__()

        assert set(hard_sharing).issubset(["input_layer","adapter","conv_block","head","batchnorm"])
        assert set(use_template_bank).issubset(["input_layer","adapter","conv_block","head"])
        self.in_shape = in_shape
        self.num_specialists = num_specialists
        self.hard_sharing = hard_sharing
        self.use_template_bank = use_template_bank
        self.average_specialists=False

        print('WRN: depth: {} , Widen Factor : {}'
              .format(depth, width))
        print('Specialists: {}'
              .format(num_specialists))
        print('Layer groups with hard sharing: {} '
              .format(hard_sharing))
        print('Layer groups using template bank: {}'
              .format(use_template_bank))

        self.tied = False
        self.bank_registry = TemplateBankRegistry()

        n_channels = [16, 16 * width, 32 * width, 64 * width]
        stride = [1, 2, 2]
        assert ((depth - 4) % 6 == 0)
        num_blocks = int((depth - 4) / 6)
        num_input_channels = in_shape[0]
        last_feature_map_shape = [in_shape[1] // 4, in_shape[2] // 4]

        self.input_layer = SConv2d(num_input_channels, n_channels[0], bank_registry=self.bank_registry,
                                   num_specialists=num_specialists,
                                   hard_sharing="input_layer" in self.hard_sharing,
                                   use_template_bank="input_layer" in self.use_template_bank,
                                   num_templates=10//compression_factor)

        self.adapter = nn.ModuleList()
        self.conv_block = nn.ModuleList()
        for i in range(3):
            self.adapter.append(Block(n_channels[i], n_channels[i + 1], stride[i], bank_registry=self.bank_registry,
                                      num_specialists=self.num_specialists,
                                      hard_sharing="adapter" in self.hard_sharing,
                                      use_template_bank="adapter" in self.use_template_bank,
                                      specialist_batchnorm=not "batchnorm" in self.hard_sharing,
                                      num_templates=(2*num_blocks-1)//compression_factor))

            blocks = []
            for _ in range(1, num_blocks):
                blocks.append(Block(n_channels[i + 1], n_channels[i + 1], 1, bank_registry=self.bank_registry,
                                    num_specialists=self.num_specialists,
                                    hard_sharing="conv_block" in self.hard_sharing,
                                    use_template_bank="conv_block" in self.use_template_bank,
                                    specialist_batchnorm=not "batchnorm" in self.hard_sharing,
                                    num_templates=(2*num_blocks-1)//compression_factor))
            if i == 2:
                blocks.append(SBatchNorm(n_channels[3], num_specialists=num_specialists,
                                         hard_sharing="batchnorm" in self.hard_sharing))

            self.conv_block.append(MultiInputSequential(*blocks))

        self.all_avgpool = nn.Sequential(nn.AvgPool2d(last_feature_map_shape),
                                     View([n_channels[3]]))

        self.head = SLinear(n_channels[3], num_classes, bank_registry=self.bank_registry,
                             num_specialists=num_specialists,
                             hard_sharing="head" in self.hard_sharing,
                             use_template_bank="head" in self.use_template_bank,
                             num_templates=10//compression_factor)

        # The order matters.
        self._init_coefficients()
        if start_tied:
            self.tie_coefficients()
        else:
            # Untied coefficients will be equal accross specialists. We therefore add noise
            # to encourage diversity.
            self._inject_diversity_noise(noise_std=noise_std_init, coef_only=True)

    def _get_layers(self):
        return [module for module in self.modules()
                          if isinstance(module, SharedLayer)]

    def _inject_diversity_noise(self, noise_std=0, coef_only=False):
        def _perturbation(source_tensor, noise_std):
            if source_tensor is None:
                return None
            # Correction term to ensure the resulting noise in the parameters will
            # have a signal to noise ratio of noise_std_init
            adjusted_noise_std = source_tensor.pow(2).mean().sqrt() * noise_std
            return torch.randn_like(source_tensor) * adjusted_noise_std
        layers = self._get_layers()
        for layer in layers:
            if layer.hard_sharing:
                continue
            if layer.coefficients is not None:
                for coefficient in layer.coefficients:
                    coefficient.data += _perturbation(coefficient, noise_std)
            elif not coef_only and layer.no_coef_modules is not None:
                for module in layer.no_coef_modules:
                    module.weight.data += _perturbation(module.weight, noise_std)
                    if module.bias is not None:
                        module.bias.data += _perturbation(module.bias, noise_std)

    def _init_coefficients(self):
        layers = self._get_layers()
        coefficient_groups_dict = defaultdict(list)
        for l in layers:
            if l.coefficients is not None:
                coefficient_groups_dict[l.bank].append(l.coefficients)
        coef_groups = coefficient_groups_dict.values()

        for coef_group in coef_groups:
            if len(coef_group) == 0:
                continue
            coefficient_groups_dict = defaultdict(list)
            for layer_coef in coef_group:
                coefficient_groups_dict[layer_coef[0].shape].append(layer_coef)
            for coef_shape, same_shape_coef_group in coefficient_groups_dict.items():
                init_value = torch.zeros(len(same_shape_coef_group),*coef_shape)
                nn.init.orthogonal_(init_value)
                for i, layer_coef in enumerate(same_shape_coef_group):
                    for specialist_coef in layer_coef:
                        with torch.no_grad():
                            specialist_coef.data = init_value[i].clone()
                            specialist_coef.data/=specialist_coef.data.norm(dim=0).mean()

    def tie_coefficients(self):
        cprint('\nTying coefficients.\n', 'red')
        if self.tied:
            warn("Coefficients already tied - not doing anything.")
            return
        layers = self._get_layers()
        for layer in layers:
            layer.tie()
        self.tied = True

    def release_coefficients(self,noise_std_init=0, reinitialize_head=False):
        cprint('\nReleasing coefficients.\n', 'red')
        if not self.tied:
            return
        layers = self._get_layers()
        for layer in layers:
            layer.release(reinitialize_head=reinitialize_head)
        self._inject_diversity_noise(noise_std=noise_std_init)
        self.tied = False

    def get_all_params(self):
        specialist_params = {"coef": [[] for _ in range(self.num_specialists)],
                             "other": [[] for _ in range(self.num_specialists)]}
        shared_params = {"coef":[], "other":[]}

        for module in self.modules():
            if isinstance(module, TemplateBank):
                shared_params["other"].append(module.templates["weight"])
                shared_params["other"].append(module.templates["bias"])
            elif isinstance(module, SharedLayer):
                if module.hard_sharing:
                    if module.use_template_bank:
                        shared_params["coef"] += [module.coefficients[0]]
                    else:
                        shared_params["other"] += [*module.no_coef_modules[0].parameters()]
                else:
                    for specialist in range(self.num_specialists):
                        if module.use_template_bank:
                            specialist_params["coef"][specialist] += [module.coefficients[specialist]]
                        else:
                            specialist_params["other"][specialist] += [*module.no_coef_modules[specialist].parameters()]

                    if module.tied:
                        if module.use_template_bank:
                            shared_params["coef"] += [module.tied_coef]
                        else:
                            shared_params["other"] += [*module.tied_module.parameters()]

        return shared_params, specialist_params

    def set_average_specialists(self, average_specialists):
        self.average_specialists=average_specialists

    def forward(self, x, specialist=None):
        assert(specialist is not None)
        x = self.input_layer(x, specialist, average_specialists=self.average_specialists)
        for i in range(3):
            x = self.adapter[i](x, specialist, average_specialists=self.average_specialists)
            x = self.conv_block[i](x, specialist, average_specialists=self.average_specialists)

        x = self.all_avgpool(nn.ReLU(inplace=True)(x))
        x = self.head(x, specialist, average_specialists=self.average_specialists)
        return x
