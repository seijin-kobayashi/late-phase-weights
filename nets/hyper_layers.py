from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from copy import deepcopy

class TemplateBankRegistry():
    """ Registry for template banks. This class ensures that all template banks for equivalent layers registerd here
        will be identical."""

    def __init__(self):
        self.registry = {}

    def _compute_key(template_bank_class, **kwargs):
        return "{}_{}".format(template_bank_class, [
                              shape for shape in template_bank_class.template_shape(**kwargs).values()])

    def get_template(self, template_bank_class, kwargs, num_templates):
        key = TemplateBankRegistry._compute_key(template_bank_class, **kwargs)
        if not key in self.registry:
            self.registry[key] = template_bank_class(num_templates, **kwargs)
        self.registry[key].init(**kwargs)
        return self.registry[key]


class TemplateBank(nn.Module, ABC):
    def __init__(self, num_templates, **kwargs):
        super(TemplateBank, self).__init__()
        self.num_templates = num_templates
        self.templates = None

    @abstractmethod
    def init(self, **kwargs):
        return NotImplementedError()

    @abstractmethod
    def template_shape(**kwargs):
        return NotImplementedError()


class SharedLayer(nn.Module, ABC):
    TEMPLATE_BANK_CLASS = None

    def __init__(self, bank_registry, num_specialists,
                 hard_sharing, use_template_bank, num_templates=None, **kwargs):
        super(SharedLayer, self).__init__()
        self.hard_sharing = hard_sharing
        self.use_template_bank = use_template_bank
        self.coefficients = None
        self.num_coefficients = 1 if hard_sharing else num_specialists
        self.tied=False
        if use_template_bank:
            assert(bank_registry is not None)
            assert(num_templates is not None)
            self.bank = bank_registry.get_template(
                self.TEMPLATE_BANK_CLASS, kwargs, num_templates=num_templates)
            self.coefficients = nn.ParameterList([nn.Parameter(self._make_coefficients())
                                                  for _ in range(self.num_coefficients)])
        else:
            self.no_coef_modules = nn.ModuleList(
                [self._make_no_coef_module() for _ in range(self.num_coefficients)])

    def tie(self):
        if self.hard_sharing:
            return
        if self.use_template_bank:
            self.tied_coef=deepcopy(self.coefficients[0])
        else:
            self.tied_module=deepcopy(self.no_coef_modules[0])

        self.tied=True

    def release(self, **kwargs):
        if self.hard_sharing:
            return
        if self.use_template_bank:
            for coef in self.coefficients:
                coef.data=self.tied_coef.clone()
        else:
            for module in self.no_coef_modules:
                module.load_state_dict(self.tied_module.state_dict())
        self.tied=False

    def _make_no_coef_module(self):
        return NotImplementedError()

    def _make_coefficients(self):
        return NotImplementedError()

    def _forward(self, input, params):
        return NotImplementedError()

    def forward(self, input, specialist, average_specialists=False):
        if average_specialists and not self.tied:
            if not self.use_template_bank:
                output=0
                for idx in range(self.num_coefficients):
                    output+=self.no_coef_modules[idx](input)
                output /= self.num_coefficients
                return output
            coefficient=0
            for idx in range(self.num_coefficients):
                coefficient+=self.coefficients[idx]
            coefficient /= self.num_coefficients
            return self._forward(input, self.bank(coefficient))

        idx = 0 if self.hard_sharing else specialist
        if not self.use_template_bank:
            module=self.no_coef_modules[idx]
            if self.tied:
                module=self.tied_module
            return module(input)
        coefficient=self.coefficients[idx]
        if self.tied:
            coefficient=self.tied_coef
        return self._forward(input, self.bank(coefficient))


class SLinear(SharedLayer):

    class LinearTemplateBank(TemplateBank):
        def init(self, in_features=None, out_features=None, **kwargs):
            self.num_templates=self.num_templates
            t_shape = self.template_shape(in_features=in_features, out_features=out_features)
            weight_templates = [torch.zeros([out_features, in_features])
                for _ in range(self.num_templates)]
            for i in range(self.num_templates):
                init.kaiming_normal_(weight_templates[i])
                weight_templates[i]=weight_templates[i].reshape(*t_shape["weight"],-1)[:,:,0]

            weight_templates = nn.Parameter(
                torch.stack(weight_templates, dim=-1))
            bias_templates = nn.Parameter(torch.zeros([*t_shape["bias"], self.num_templates]))
            self.templates = nn.ParameterDict(
                {"weight": weight_templates, "bias": bias_templates})

        @staticmethod
        def template_shape(in_features=None, out_features=None, **kwargs):
            return {"weight": [out_features, in_features], "bias":[out_features]}

        def forward(self, coef):
            weight = torch.matmul(self.templates["weight"], coef)
            weight = weight.permute([0,2,1]).reshape(weight.shape[0],-1)
            bias = torch.matmul(self.templates["bias"], coef).sum(dim=1)
            return [weight, bias]

    TEMPLATE_BANK_CLASS = LinearTemplateBank

    def __init__(self, in_features, out_features, bank_registry=None,
                 num_specialists=1, hard_sharing=False, use_template_bank=False,
                 num_templates=None):
        self.in_features = in_features
        self.out_features = out_features

        super(SLinear, self).__init__(bank_registry, num_specialists, hard_sharing,
                                      use_template_bank, in_features=in_features,
                                      out_features=out_features,
                                      num_templates=num_templates)

    def release(self, reinitialize_head=False):
        if self.hard_sharing or not reinitialize_head:
            return super(SLinear, self).release()
        else:
            if self.use_template_bank:
                for i in range(self.bank.templates['weight'].shape[-1]):
                    init.kaiming_normal_(self.bank.templates['weight'][:,:,i])
            else:
                for module in self.no_coef_modules:
                    module.load_state_dict(self._make_no_coef_module().state_dict())
            self.tied=False

    def _make_no_coef_module(self):
        linear = nn.Linear(self.in_features, self.out_features)
        init.kaiming_normal_(linear.weight)
        linear.bias.data.zero_()
        return linear

    def _make_coefficients(self):
        return torch.zeros([self.bank.num_templates, 1])

    def _forward(self, input, params):
        return F.linear(input, params[0], params[1])


class SBatchNorm(SharedLayer):
    def __init__(self, num_features, num_specialists=1,
                 hard_sharing=False):
        self.num_features=num_features
        super(SBatchNorm, self).__init__(None, num_specialists,
                                         hard_sharing, False, num_features=num_features)

    def _make_no_coef_module(self):
        batchnorm = nn.BatchNorm2d(self.num_features)
        batchnorm.weight.data.fill_(1)
        batchnorm.bias.data.zero_()
        return batchnorm


class SConv2d(SharedLayer):
    class Conv2dTemplateBank(TemplateBank):
        def init(self, in_channels=None, out_channels=None, kernel_size=3, bias=False,
                 template_split_factor=1, **kwargs):
            self.bias=bias

            t_shape = self.template_shape(in_channels=in_channels,
                                                    out_channels=out_channels,
                                                    kernel_size=kernel_size,
                                                    template_split_factor=template_split_factor)

            templates = [torch.zeros([out_channels, in_channels, kernel_size, kernel_size])
                for _ in range(self.num_templates)]

            for i in range(self.num_templates):
                init.kaiming_normal_(templates[i])
                templates[i]=templates[i].reshape(*t_shape["weight"],-1)[:,:,:,:,0]
                templates[i]=templates[i].permute([0,2,3,1])

            weight_templates = nn.Parameter(torch.stack(templates, dim=4))
            bias_templates = nn.Parameter(torch.zeros([*t_shape["bias"], self.num_templates]))
            self.templates = nn.ParameterDict(
                {"weight": weight_templates, "bias": bias_templates})

        @staticmethod
        def template_shape(in_channels=None, out_channels=None,
                           kernel_size=3, template_split_factor=1, **kwargs):
            return {"weight": [out_channels,
                               in_channels//template_split_factor,
                               kernel_size, kernel_size], "bias":[out_channels]}

        def forward(self, coef):
            weight = torch.matmul(self.templates["weight"], coef)
            weight = weight.view(*(weight.shape[:3]),-1).permute([0, 3, 1, 2])
            bias = torch.matmul(self.templates["bias"], coef).sum(dim=1) if self.bias else None
            return [weight, bias]

    TEMPLATE_BANK_CLASS = Conv2dTemplateBank

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False,
                 bank_registry=None, num_specialists=1, hard_sharing=False,
                 use_template_bank=False, template_split_factor=1,
                 num_templates=None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.template_split_factor=template_split_factor
        assert(in_channels % template_split_factor == 0)
        super(SConv2d, self).__init__(bank_registry, num_specialists, hard_sharing,
                                      use_template_bank,
                                      in_channels=in_channels,
                                      out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, bias=bias,
                                      template_split_factor=template_split_factor,
                                      num_templates=num_templates)

    def _make_no_coef_module(self):
        conv = nn.Conv2d(self.in_channels, self.out_channels,
                  kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.bias)
        init.kaiming_normal_(conv.weight)

        if self.bias:
            init.constant_(conv.bias, 0)
        return conv

    def _make_coefficients(self):
        return torch.zeros(
            [self.bank.num_templates, self.template_split_factor])

    def _forward(self, input, params):
        return F.conv2d(input, params[0], bias=params[1], stride=self.stride,
                        padding=self.padding)
