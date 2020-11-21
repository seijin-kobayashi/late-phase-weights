#!/usr/bin/env python3

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import torch
from tensorboardX import SummaryWriter
import numpy as np
import json
import copy
from termcolor import cprint
import dill

from utils.train_test import hyper_train, test, test_ood
from utils.model_utils import setup_ensemble, losses
from utils.data_utils import load_data
from utils.shell_utils import configure_display, parse_shell_args, update_config
from utils import optim_utils
import pickle
import shutil

_SUMMARY_KEYWORDS = [
    # Track all performance measures with respect to the best mean accuracy.
    'num_parameters',
    'num_coefs',
    'final_test_acc',
    'final_nll',
    'final_auroc',
    'num_train_epochs',
    'finished'
]

def list_to_str(list_arg, delim=' '):
    ret = ''
    for i, e in enumerate(list_arg):
        if i > 0:
            ret += delim
        ret += str(e)
    return ret

def save_performance_summary(config, train_epochs=None, training_finished=False):
    # save some stuff in a pickle for later
    print("Saving performance summary to {}".format(os.path.join(config["out_dir"],'performance_overview.txt')))
    with open(config["out_dir"] + "/training_results.pickle", "wb") as f:
                pickle.dump([config['auroc_dict'], config['test_acc_dict'],
                         config['test_loss_dict']], f, pickle.HIGHEST_PROTOCOL)

    if train_epochs is None:
        train_epochs = config["epochs"]

    tp = dict()
    tp["num_parameters"] = config["num_parameters"]
    tp["num_coefs"] = config["num_coefs"]
    tp["final_test_acc"] = str(config["final_test_acc"])
    tp["final_nll"] = str(config['test_loss_dict'][config['epochs']])
    tp["final_auroc"] = list_to_str(config["final_auroc"])
    tp["finished"] = 1 if training_finished else 0

    with open(os.path.join(config["out_dir"],'performance_overview.txt'), 'w') as f:
        for kw in _SUMMARY_KEYWORDS:
            if kw == 'num_train_epochs':
                f.write('%s %d\n' % ('num_train_epochs', train_epochs))
                continue
            else:
                try:
                    f.write('%s %f\n' % (kw, tp[kw]))
                except:
                    f.write('%s %s\n' % (kw, tp[kw]))

def setup_environment(config):
    ### Deterministic computation.
    if config['random_seed'] == -1:
        config["random_seed"] = np.random.randint(2**32)
    torch.manual_seed(config['random_seed'])
    torch.cuda.manual_seed_all(config['random_seed'])
    np.random.seed(config['random_seed'])

    ### Output folder.
    if os.path.exists(config["out_dir"]):
        response = input('The output folder %s already exists. ' % \
                         (config["out_dir"]) + \
                         'Do you want us to delete it? [y/n]')
        if response != 'y':
            raise Exception('Could not delete output folder!')
        shutil.rmtree(config["out_dir"])

        os.makedirs(config["out_dir"])
        print("Created output folder %s." % (config["out_dir"]))

    else:
        os.makedirs(config["out_dir"])
        print("Created output folder %s." % (config["out_dir"]))

    # Save user configs to ensure reproducibility of this experiment.
    with open(os.path.join(config["out_dir"], 'config.pickle'), 'wb') as f:
        pickle.dump(config, f)
    # A JSON file is easier to read for a human.
    with open(os.path.join(config["out_dir"], 'config.json'), 'w') as f:
        json.dump(config, f)

    log_file_name = config['log_file'] + str(os.getpid()) + ".txt"
    if not os.path.exists(os.path.dirname(log_file_name)):
        os.makedirs(os.path.dirname(log_file_name))
    log_file = open(log_file_name, 'w')

    # Read config to figure out whether to use CUDA.
    use_cuda = config['cuda'] and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.fastest = True
    if config['save_path'] == "None":
        config['save_path'] = config['out_dir'] + "/"
    if not os.path.exists(config["save_path"]):
        os.makedirs(config["save_path"])

    if config['use_tensorboard']:
        writer = SummaryWriter(log_dir=config["out_dir"])
        config['TB_cwd'] = writer.file_writer.get_logdir
    else:
        writer = None

    return log_file, device, writer

if __name__ == '__main__':
    configure_display()

    # Parse shell arguments as input configuration.
    user_config = parse_shell_args(sys.argv[1:])

    # Load base configuration file.    
    with open(os.path.join(os.path.dirname(__file__), 'etc', user_config['config']+'.json')) as config_json_file:
        config = json.load(config_json_file)

    # First update with argparse, needed to get load and save, etc.
    update_config(config, user_config)

    log_file, device, writer = setup_environment(config)

    # Print out configuration
    print('PID: {}'.format(os.getpid()))

    config['auroc_dict'] = {}
    config['test_acc_dict'] = {}
    config['test_loss_dict'] = {}
    ######### Load datasets.
    train_loader, test_loader, ood_test_loader_list = load_data(config)
    minimal_ood_test_loader_list = [ood_test_loader_list[0]]

    ######### Main training and testing loop.
    print('Training model...')

    task_loss_criterion = losses[config['loss']]
    ensemble, model, optimizers, schedulers = setup_ensemble(config, device, train_loader)

    model_to_test = model
    ensemble_to_test = ensemble

    for epoch in range(1, config['epochs'] + 1):
        print('Epoch: {}'.format(epoch))

        print('Training main network...')

        model.train()
        model.set_average_specialists(False)
        cprint('Model learning rates\nbase: {}'.format(optimizers.get_base_lr_str()), 'blue')
        cprint('coef: {}\n'.format(optimizers.get_coef_lr_str()), 'blue')

        ## Train the model.
        hyper_train(config, log_file,
            train_loader, device,
            model, optimizers,
            task_loss_criterion, epoch,
            log=True, writer=writer,
            inner_loop_training=config['inner_training'])

        ## Test and validate the model.
        if config['optimizer'] == 'SWA' and epoch >= config['swa_start']:
            if epoch == config['swa_start']:
                swa_ensemble = copy.deepcopy(ensemble)
                swa_model = swa_ensemble.get_models()
                swa_n = 0

            # Swap models to be tested, from the SGD-trained model
            # to the averaged one.
            model_to_test = swa_model
            ensemble_to_test = swa_ensemble

            # Update SWA model parameters, including BN statistics.
            cprint('Updating SWA model', 'red')
            optim_utils.moving_average(
                swa_model,
                model,
                1.0 / (swa_n + 1))
            optim_utils.bn_update(
                train_loader,
                swa_model)
            swa_n += 1

        ensemble_to_test.eval()
        model_to_test.set_average_specialists(True)

        # Step every scheduler, for every model.
        if config['lr_scheduler']:
            for scheduler in schedulers:
                scheduler.step()

        # Compute test set accuracy using ensemble-averaged predictions.
        print('Testing model...')
        test_acc = test(config, log_file, test_loader,
                        ensemble_to_test, device, epoch,
                        writer=writer, compute_nll=True)
        config['test_acc_dict'][epoch] = test_acc

        # Calculate OOD statistics using ensemble-averaged entropies.
        if "ood" in config['problem'] and (epoch - 1) % config['ood_log_interval'] == 0:
            aurocs = test_ood(config, log_file, test_loader,
                              minimal_ood_test_loader_list,
                              ensemble_to_test,
                              device, epoch, writer=writer)
            config['auroc_dict'][epoch] = aurocs

    config['final_test_acc']=test_acc
    print('Done training on all tasks.')
    if "ood" in config['problem']:
        final_auroc = test_ood(config, log_file, test_loader,
                                ood_test_loader_list, ensemble_to_test,
                                device, epoch, writer=writer)
        config['final_auroc'] = final_auroc
        print("Final AUROC averaged over datasets: ", np.mean(final_auroc))
    print("Final test set accuracy: ", test_acc)
    
    save_performance_summary(config, training_finished=True)

    final_model_path = config['save_path'] + 'final_' + config["run_id"] \
            + str(os.getpid()) + ".pth.tar"
    print('Saving final checkpoint at {}.'.format(final_model_path))

    checkpoint = {
        'epoch': epoch,
        'ensembles': ensemble,
        'model': model,
        'optimizers': optimizers,
        'schedulers': schedulers,
#        'config': config
    }

    if config['optimizer'] == 'SWA' and epoch > config['swa_start']:
        checkpoint['swa_ensemble'] = swa_ensemble
        checkpoint['swa_model'] = swa_model
        checkpoint['swa_n'] = swa_n

    torch.save(checkpoint, final_model_path, pickle_module=dill)

    writer.close()
    print("Finished Training - check results in: ", config['out_dir'])
