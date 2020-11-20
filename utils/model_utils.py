#!/usr/bin/env python3
# Please do not redistribute.

import torch.nn as nn
import numpy as np
import nets.models
from utils.optim_utils import OptimizerSet, create_scheduler, create_optimizer


losses = nn.ModuleDict([
    ['MSE', nn.MSELoss(reduction='mean')],
    ['BCE', nn.BCELoss(reduction='mean')],
    ['CE', nn.CrossEntropyLoss(reduction='mean')],
    ['NLLLoss', nn.NLLLoss(reduction='mean')]])


def setup_ensemble(config, device, train_loader):
    specialist_list = []

    model = nets.models.HYPERWRN(
        depth=config['wrn_depth'],
        width=config['wrn_width'],
        num_classes=config['num_classes_per_task'],
        start_tied=config['coef_start_tied'],
        in_shape=[3,32,32],
        num_specialists=config['num_specialists'],
        noise_std_init=config['noise_std_init'],
        hard_sharing = config["hard_sharing"],
        use_template_bank = config["use_template_bank"],
        )

    model = model.to(device)
    if config['verbose']: print(model)

    specialist_list = model
    # Gather parameters, handling coefficients separately.
    coef_p_list = []

    base_p_list = []
    num_parameters_coefs = 0
    num_parameters_templates = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            if 'coefficients' in name:
                num_parameters_coefs += np.prod((p.shape))
                coef_p_list += [p]
            else:
                num_parameters_templates += np.prod((p.shape))
                base_p_list += [p]

    print('Number of trainable parameters in the network: ',
        num_parameters_coefs + num_parameters_templates)

    print('Number of coefficients: {} - ({:.2f} %)'.format(
        num_parameters_coefs,
        100*num_parameters_coefs/(num_parameters_coefs +
            num_parameters_templates)))

    config["num_parameters"] = int(num_parameters_coefs
        + num_parameters_templates)
    config["num_coefs"] = int(num_parameters_coefs)

    # Create and configure optimizers and schedulers for current model.
    scheduler_list = []
    optimizer_dict = {}
    optimizer_dict['shared'] = {}
    optimizer_dict['specialist'] = {}
    optimizer_dict['specialist']['coef'] = []
    optimizer_dict['specialist']['other'] = []


    all_params = model.get_all_params()
    all_shared_p = all_params[0]
    all_specific_p = all_params[1]

    coef_shared_p = all_shared_p['coef']
    other_shared_p = all_shared_p['other']

    coef_specific_p = all_specific_p['coef']
    other_specific_p = all_specific_p['other']

    for s in range(config['num_specialists']):
        if coef_specific_p[s]:
            params = coef_specific_p[s]
            optimizer = create_optimizer(config, params, train_loader,
                                         coef=True, specialist=True)
            optimizer_dict['specialist']['coef'] += [optimizer]

            if config['lr_scheduler']:
                scheduler = \
                    create_scheduler(config, optimizer, params,
                                     coef=True)
                scheduler_list += [scheduler]

        else:
            optimizer_dict['specialist']['coef'] = None

        if other_specific_p[s]:
            params = other_specific_p[s]
            optimizer = create_optimizer(config, params, train_loader,
                                         coef=False, specialist=True)
            optimizer_dict['specialist']['other'] += [optimizer]

            if config['lr_scheduler']:
                scheduler = \
                    create_scheduler(config, optimizer, params,
                                     coef=False)
                scheduler_list += [scheduler]
        else:
            optimizer_dict['specialist']['other'] = None

    if other_shared_p:
        params = other_shared_p
        optimizer = create_optimizer(config, params, train_loader,
                                     coef=False, specialist=False)
        optimizer_dict['shared']['other'] = optimizer

        if config['lr_scheduler']:
            scheduler = \
                create_scheduler(config, optimizer, params, coef=False)
            scheduler_list += [scheduler]
    else:
        optimizer_dict['shared']['other'] = None

    if coef_shared_p:
        params = coef_shared_p
        optimizer = create_optimizer(config, params, train_loader,
                                     coef=True, specialist=False)
        optimizer_dict['shared']['coef'] = optimizer

        if config['lr_scheduler']:
            scheduler = \
                create_scheduler(config, optimizer, params, coef=True)
            scheduler_list += [scheduler]
    else:
        optimizer_dict['shared']['coef'] = None

    optimizer_set = OptimizerSet(
        shared_p_optimizers=optimizer_dict['shared'],
        specific_p_optimizers=optimizer_dict['specialist'],
        legacy_mode=False)

    ensemble = nets.models.VirtualEnsemble(specialist_list)
    ensemble = ensemble.to(device)

    return ensemble, specialist_list, optimizer_set, scheduler_list
