#!/usr/bin/env python3
# Please do not redistribute.

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import numpy as np
from datetime import datetime
import argparse
from utils.argparse_embelishment import ArgumentParser

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def configure_display():
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)

def maybe_split_string_to_list(config, key, cast=None):
    if key not in config or isinstance(config[key], list):
        # Do nothing.
        return
    config[key] = config[key].split(",") if config[key] != "" else []
    config[key] = [hs.strip() for hs in config[key]]
    if cast is not None:
        config[key] = [cast(hs) for hs in config[key]]

def update_config(config, user_config):
    # Do the main updating, based on the user_config extracted by argparse.
    config.update(user_config)  

    if config["out_dir"] == 'out/':
        run_id=config["run_id"]+"_" if "run_id" in config else ""
        config["out_dir"] += run_id + 'run_'+datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    config['data_folder'] = os.path.join(os.path.dirname(__file__), '../datasets')

    config['PID'] = os.getpid()

    # Update some configuration options for later ease of manipulation.
    maybe_split_string_to_list(config, "lr_multistep_milestones", cast=int)
    maybe_split_string_to_list(config, "coef_lr_multistep_milestones", cast=int)
    maybe_split_string_to_list(config, "hard_sharing")
    maybe_split_string_to_list(config, "use_template_bank")

    if "test_loss_dict" not in config.keys():
        config['test_loss_dict'] = {}

    if "auroc_dict" not in config.keys():
        config['auroc_dict'] = {}

    if "test_acc_dict" not in config.keys():
        config['test_acc_dict'] = {}

    # Run some sanity checks.
    assert(config['coef_release_epoch'] < config['swa_start'])
    
    for opt in config["hard_sharing"]:
        assert(opt in ["input_layer", "adapter", "conv_block", "head",
                    "batchnorm"])
    
    for opt in config["use_template_bank"]:
        assert(opt in ["input_layer", "adapter", "conv_block", "head"])


def parse_shell_args(args):
    parser = ArgumentParser(
            description="Run task inference experiment.")

    parser.add_argument("--config",
                            choices=["cifar10_bn", "cifar10_bn_swa", "cifar100_bn", "cifar100_bn_swa", "cifar100_hnet", "cifar100_hnet_swa"],
                            default=argparse.SUPPRESS,
                            help="Configuration to run.")

    parser.add_argument("--problem",
                            choices=["oodCIFAR10", "oodCIFAR100"],
                            default=argparse.SUPPRESS,
                            help="Problem to solve.")
    parser.add_argument("--random-seed",
                            type=int,
                            default=argparse.SUPPRESS,
                            help="Random seed used for numpy and torch.")                 
    parser.add_argument("--out-dir",
                            default=argparse.SUPPRESS,
                            help="Directory to save output files, checkpoints.")

    parser.add_argument("--run-id",
                        default=argparse.SUPPRESS,
                        help="Name of the run.")

    parser.add_argument("--num-tasks",
                            type=int,
                            default=argparse.SUPPRESS,
                            help="Number of CIFAR tasks.")

    parser.add_argument("--num-classes-per-task",
                            type=int,
                            default=argparse.SUPPRESS,
                            help="Number of classes per CIFAR task.")

    parser.add_argument("--augment-data",
                            type=str2bool,
                            nargs='?',
                            const=True,
                            default=argparse.SUPPRESS,
                            help="Turn on CIFAR data augmentation?")

    parser.add_argument("--cutout-augment",
                            type=str2bool,
                            nargs='?',
                            const=True,
                            default=argparse.SUPPRESS,
                            help="Use cutout augmentation during training?")
    
    parser.add_argument("--random-erasing",
                            type=str2bool,
                            nargs='?',
                            const=True,
                            default=argparse.SUPPRESS,
                            help="Use random erasing augmentation during training?")

    parser.add_argument("--dimensions",
                            type=int,
                            nargs='+',
                            default=argparse.SUPPRESS,
                            help="Network dimensions specified in the order: "
                            + " input, hidden layers, output."
                            + "Applies only to fcnet.")

    parser.add_argument("--verbose", 
                            action='store_true',
                            default=argparse.SUPPRESS,
                            help="See more print outs.")

    parser.add_argument("--wrn-depth",
                            type=int,
                            default=argparse.SUPPRESS,
                            help="Depth of the WRN ResNet.")

    parser.add_argument("--wrn-width",
                            type=int,
                            default=argparse.SUPPRESS,
                            help="Width of the WRN ResNet layers.")

    parser.add_argument("--loss",
                            choices=["CE", "MSE"],
                            default=argparse.SUPPRESS,
                            help="Task loss.")

    parser.add_argument("--batch-size",
                            type=int,
                            default=argparse.SUPPRESS,
                            help="Training batch size.")

    parser.add_argument("--test-batch-size",
                            type=int,
                            default=argparse.SUPPRESS,
                            help="Testing batch size.")

    parser.add_argument("--epochs",
                            type=int,
                            default=argparse.SUPPRESS,
                            help="Training epochs.")

    parser.add_argument("--cuda",
                            type=str2bool,
                            nargs='?',
                            const=True,
                            default=argparse.SUPPRESS,
                            help="Use CUDA backend?")

    parser.add_argument("--seed",
                            type=float,
                            default=argparse.SUPPRESS,
                            help="Pytorch random seed.")

    parser.add_argument("--log-interval",
                            type=int,
                            default=argparse.SUPPRESS,
                            help="Logging interval (in # of batches).")

    parser.add_argument("--ood-log-interval",
                            type=int,
                            default=argparse.SUPPRESS,
                            help="OOD logging interval (in # of epochs).")

    parser.add_argument("--individual-test-interval",
                            type=int,
                            default=argparse.SUPPRESS,
                            help="Specialist testing interval (in # of epochs).")

    parser.add_argument("--log-file",
                            type=str,
                            default=argparse.SUPPRESS,
                            help="Log file path prefix.")

    parser.add_argument("--use-tensorboard",
                            type=str2bool,
                            nargs='?',
                            const=True,
                            default=argparse.SUPPRESS,
                            help="Use tensorboard?")

    parser.add_argument("--data-folder",
                            type=str,
                            default=argparse.SUPPRESS,
                            help="Path to dataset storage folder.")

    parser.add_argument("--num-specialists",
                            type=int,
                            default=argparse.SUPPRESS,
                            help="Number of specialists (heads) in one ResNet."+
                                 "For CIFAR you can choose from the following "+
                                 " list: [1,3,5,8,10,16,32,64]. This has no " +
                                 "effect if multi_head ==random_head == False.")

    #######################
    ## SGD & Adam arguments 
    #######################
    parser.add_argument("--optimizer",
                            choices=["SGD", "Adam", "SWA"],
                            default=argparse.SUPPRESS,
                            help="Specialist optimizer.")

    parser.add_argument("--lr",
                            type=float,
                            default=argparse.SUPPRESS,
                            help="(Initial) learning rate.")

    parser.add_argument("--coef-lr",
                            type=float,
                            default=argparse.SUPPRESS,
                            help="(Initial) coefficient learning rate.")

    parser.add_argument("--specialist-lr",
                            type=float,
                            default=argparse.SUPPRESS,
                            help="(Initial) specialist learning rate.")

    parser.add_argument("--coef-specialist-lr",
                            type=float,
                            default=argparse.SUPPRESS,
                            help="(Initial) coefficient specialist learning rate.")

    parser.add_argument("--nesterov",
                            type=str2bool,
                            nargs='?',
                            const=True,
                            default=argparse.SUPPRESS,
                            help="Whether Nesterov momentum is used"+
                                 "with SGD.")

    parser.add_argument("--momentum",
                            type=float,
                            default=argparse.SUPPRESS,
                            help="Momentum for SGD.")

    parser.add_argument("--weight-decay",
                            type=float,
                            default=argparse.SUPPRESS,
                            help="Weight decay for SGD.")

    parser.add_argument("--coef-weight-decay",
                            type=float,
                            default=argparse.SUPPRESS,
                            help="Coefficient weight decay for SGD.")

    parser.add_argument("--specialist-weight-decay",
                            type=float,
                            default=argparse.SUPPRESS,
                            help="Specialist weight decay for SGD.")

    parser.add_argument("--coef-specialist-weight-decay",
                            type=float,
                            default=argparse.SUPPRESS,
                            help="Coefficient specialist weight decay for SGD.")

    parser.add_argument("--lr-scheduler",
                            choices=["annealing", "multistep"],
                            default=argparse.SUPPRESS,
                            help="Learning rate scheduler.")

    parser.add_argument("--lr-multistep-gamma",
                            type=float,
                            default=argparse.SUPPRESS,
                            help="Learning rate adjustment factor.")

    parser.add_argument("--lr-multistep-milestones",
                            type=str,
                            default=argparse.SUPPRESS,
                            help="Learning rate adjustment times (in epochs)."+
                                 " Adjustment occurs after said epoch.")

    parser.add_argument("--coef-lr-multistep-gamma",
                            type=float,
                            default=argparse.SUPPRESS,
                            help="Coefficient learning rate adjustment factor.")

    parser.add_argument("--coef-lr-multistep-milestones",
                            type=str,
                            default=argparse.SUPPRESS,
                            help="Coefficient learning rate adjustment times"
                            + " (in epochs).")

    parser.add_argument("--inner-training",
                            type=str2bool,
                            nargs='?',
                            const=True,
                            default=argparse.SUPPRESS,
                            help="Update metamodel weights only once every "+
                                 "specialist has been updated.")

    parser.add_argument("--freeze-metamodel-epoch",
                            type=int,
                            default=argparse.SUPPRESS,
                            help="Epoch after which metamodel is frozen.")


    parser.add_argument("--swa-start",
                            type=int,
                            default=argparse.SUPPRESS,
                            help="Epoch (beginning of which) SWA starts.")


    #######################
    ## HYPERWRN
    #######################
    parser.add_argument("--noise-std-init",
                            type=float,
                            default=argparse.SUPPRESS,
                            help="Standard deviation of noise, for noisy init"
                                 + " or noisy coefficient release.")

    parser.add_argument("--coef-start-tied",
                            type=str2bool,
                            nargs='?',
                            const=True,
                            default=argparse.SUPPRESS,
                            help="Whether specialist coefficients start tied.")

    parser.add_argument("--coef-release-epoch",
                            type=float,
                            default=argparse.SUPPRESS,
                            help="If coefficients start tied, epoch to release.")

    parser.add_argument("--coef-release-init",
                            choices=["equal", "orthogonal"],
                            default=argparse.SUPPRESS,
                            help="Coefficient release strategy.")

    parser.add_argument("--specialist-batchnorm",
                            type=str2bool,
                            nargs='?',
                            const=True,
                            default=argparse.SUPPRESS,
                            help="Use a dedicated batchnorm unit for each "+
                                 "specialist?")
    
    parser.add_argument("--annealing-end-lr",
                            type=float,
                            default=argparse.SUPPRESS,
                            help="Learning rate after annealing " +
                                 "had happend.")

    parser.add_argument("--coef-annealing-end-lr",
                            type=float,
                            default=argparse.SUPPRESS,
                            help="Learning rate of the coefficients after " +
                                 "annealing had happend.")
    
    parser.add_argument("--annealing-end",
                            type=int,
                            default=argparse.SUPPRESS,
                            help="End of lr annealing (# epochs).")

    parser.add_argument("--coef-annealing-end",
                            type=int,
                            default=argparse.SUPPRESS,
                            help="End of coefficient lr annealing (# epochs).")
    
    parser.add_argument("--annealing-start",
                            type=int,
                            default=argparse.SUPPRESS,
                            help="Start of lr annealing (# epochs).")

    parser.add_argument("--coef-annealing-start",
                            type=int,
                            default=argparse.SUPPRESS,
                            help="Start of coefficient lr annealing (# epochs).")

    parser.add_argument("--hard-sharing",
                            type=str,
                            default=argparse.SUPPRESS,
                            help="Comma separated str indicating which part " +
                                "of the network is shared across the ensemble.")

    parser.add_argument("--use-template-bank",
                            type=str,
                            default=argparse.SUPPRESS,
                            help="Comma separated str indicating which part " +
                                "of the network is modeled by a hypernetwork.")

    parser.add_argument("--reinitialize-head-at-release",
                            type=str2bool,
                            nargs='?',
                            const=True,
                            default=argparse.SUPPRESS,
                            help="Use standard init for the head when releasing.")

   
    #######################
    ## Debugging
    #######################
    parser.add_argument("--extra-verbose",
                            type=str2bool,
                            nargs='?',
                            const=True,
                            default=argparse.SUPPRESS,
                            help="Do even more checking and printing.")


    #######################
    ## Checkpointing
    #######################
    parser.add_argument("--save-path",
                            type=str,
                            default=argparse.SUPPRESS,
                            help="Where to save model checkpoint.")

    #######################
    ## TO DELETE
    #######################

    parser.add_argument("--break-batch-idx",
                            type=int,
                            default=argparse.SUPPRESS,
                            help="Break epoch after specified # of batches.")

    parser.add_argument("--detailed-hnet-analysis-interval",
                            type=int,
                            default=argparse.SUPPRESS,
                            help="")
                            

    return vars(parser.parse_args(args))
