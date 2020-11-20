#!/usr/bin/env python3
# Please do not redistribute.

import torch
import torch.nn as nn
import torch.nn.functional as F
import nets.hyper_wrn

class BaseModel():
    def supports_multihead(self):
        return False

    def has_coefficients(self):
        return False

    def softmax_output(self):
        return False

    def entropy(self, x, eps=1e-5):
        y = self(x)
        p = F.softmax(y, dim=1)
        entropy = - torch.sum(p * torch.log(p + eps), dim=1)
        return entropy

class HYPERWRN(nets.hyper_wrn.HyperWRN, BaseModel):
    def __init__(self, depth=28,
                    width=10,
                    num_classes=10,
                    num_specialists=1,
                    noise_std_init=0,
                    start_tied=False,
                    in_shape=[3, 32, 32],
                    hard_sharing=[
                    "input_layer",
                    "adapter",
                    "conv_block",
                    "head",
                    "batchnorm"],
                    use_template_bank=[
                    "input_layer", "adapter", "conv_block", "head"]):
        
        super(HYPERWRN, self).__init__(
            depth,
            width,
            num_specialists=num_specialists,
            num_classes=num_classes,
            in_shape=in_shape,
            hard_sharing=hard_sharing,
            use_template_bank=use_template_bank,
            start_tied=start_tied,
            noise_std_init=noise_std_init)
    
    def supports_multihead(self):
        return True

    def has_coefficients(self):
        return True
    
    def entropy(self, x, specialist=None, eps=1e-5):
        if specialist is not None:
            y = self(x, specialist=specialist)
        else:
            y = self(x)
        
        p = F.softmax(y, dim=1)
        entropy = - torch.sum(p * torch.log(p + eps), dim=1)

        return entropy

class DeepEnsemble(nn.Module):
    def __init__(self, models):
        super().__init__()

        self.models = models
        self.T = nn.Parameter(torch.ones(1))

    def get_models(self):
        return self.models

    def query_all(self, x):
        y = []
        for i, model in enumerate(self.models):
            y += [model(x)]
        y = torch.stack(y)
        return y

    def temperature_scale(self, logits):
        return logits / self.T

    def entropy(self, x, eps=1e-5):
        y = self.query_all(x)
        y = self.temperature_scale(y)
        p = F.softmax(y, dim=2)
        avg_p = torch.mean(p, dim=0)
        entropy = - torch.sum(avg_p * torch.log(avg_p + eps), dim=1)
        return entropy

    def softmax_output(self):
        return True
        
    # Note: already returns softmax!
    def forward(self, x, specialist=None):
        if specialist is not None:
            model = self.models[specialist]
            y = model(x)
            y = self.temperature_scale(y)
            p = F.softmax(y, dim=1)
            return p
        else:
            y = self.query_all(x)
            y = self.temperature_scale(y)
            p = F.softmax(y, dim=2)
            avg_p = torch.mean(p, dim=0)
            return avg_p

class VirtualEnsemble(DeepEnsemble):
    def __init__(self, models):
        super().__init__(models)
    
    def num_specialists(self):
        return self.models.num_specialists

    def query_specialist(self, x, specialist):
        model = self.models
        y = model(x, specialist=specialist)
        return y

    def entropy(self, x, specialist=None, eps=1e-5):
        model = self.models
        if specialist is not None:
            return model.entropy(x, specialist=specialist, eps=eps)
        else:
            return super().entropy(x, eps=eps)

    def query_all(self, x):
        y = []
        model = self.models
        for i in range(model.num_specialists):
            y += [model(x, specialist=i)]
        y = torch.stack(y)
        return y

    def forward(self, x, specialist=None):
        if specialist is not None:
            model = self.models
            y = model(x, specialist=specialist)
            y = self.temperature_scale(y)
            p = F.softmax(y, dim=1)
            return p
        else:
            return super().forward(x)
