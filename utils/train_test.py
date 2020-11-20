#!/usr/bin/env python3
# Please do not redistribute.

import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
from termcolor import cprint
from sklearn.metrics import roc_auc_score

def hyper_train(config,
                log_file,
                train_loader,
                device,
                model,
                optimizer_set,
                loss_criterion,
                epoch,
                log=True,
                writer=None,
                inner_loop_training=False):

    n_batches_per_epoch = len(train_loader)
    num_vs = config['num_specialists']

    # Check whether coefficients need to be released.
    # Plus 1 ensures that the coefficients are released at the end of said epoch.
    # This in turn ensures compatibility with the scheduler logic of pytorch.
    if (config['coef_start_tied']
        and epoch == config['coef_release_epoch'] + 1):
        model.release_coefficients(
            noise_std_init=config['noise_std_init'],
            reinitialize_head=config['reinitialize_head_at_release'])
	
    coef_have_been_released = (not config['coef_start_tied']	
                                or (epoch >= config['coef_release_epoch'] + 1))

    if epoch == (config['freeze_metamodel_epoch'] + 1):
        cprint('\nFreezing metamodel.\n', 'red')

    if inner_loop_training: curr_vs = 0
    update_shared_model = True
    n_samples = 0

    for batch_idx, (x, t) in enumerate(train_loader):
        if batch_idx == config['break_batch_idx']:
            break
        
        current_batch_size = x.data.size()[0]
        n_samples += current_batch_size

        x = x.to(device)

        if config['loss'] == 'CE' or config['loss'] == 'NLLLoss':
            if config['cuda']:
                target = t.to(device, non_blocking=True)
            else:
                target = t.to(device)
        else:
            raise Exception('Currently unsupported loss.')
        
        if inner_loop_training:
            # Always zero-out specialist-specific gradients.
            optimizer_set.specific_p_zero_grad()
            
            # Only zero-out shared model gradients if we already cycled through
            # every specialist.
            if (curr_vs == 0) or (not coef_have_been_released):
                optimizer_set.shared_p_zero_grad()
            
            if (curr_vs == num_vs - 1) or (not coef_have_been_released):
                update_shared_model = True
            else:
                update_shared_model = False
            specialist_idx = curr_vs
            # Advance loop through specialists.
            curr_vs = (curr_vs + 1) % num_vs

        else:
            # Always zero-out gradients for every parameter here.
            optimizer_set.zero_grad()
            # Randomized specialist training.
            specialist_idx = np.random.randint(num_vs)

        y = model(x, specialist=specialist_idx)
        loss = loss_criterion(y, target)

        # Backpropagate errors.
        loss.backward()

        # Take a step (respecting inner-loop learning step).
        # OptimizerSet takes care of stepping only for the current specialist.
        optimizer_set.specific_p_step(specialist_idx=specialist_idx)
        if update_shared_model and (epoch < (config['freeze_metamodel_epoch']+1)):
            optimizer_set.shared_p_step()
        # Write tensorboard and log.
        with torch.no_grad():
            if writer is not None and batch_idx % 100 == 0:
                curr_step = config['batch_size'] \
                    * ((epoch - 1) * n_batches_per_epoch + batch_idx)

                writer.add_scalar('loss', loss.item(), curr_step)

                shared_p = optimizer_set.get_shared()
                p_count = 0
                norm_sum = 0.
                for p in shared_p:
                    if p.grad is not None:
                        norm_sum += torch.sum(torch.abs(p.grad.data))
                        p_count += np.array(p.grad.shape).sum()

                if p_count > 0:
                    shared_p_avg_grad_norm = norm_sum / p_count
                else:
                    shared_p_avg_grad_norm = 0

                writer.add_scalar('param_norm_shared',
                    shared_p_avg_grad_norm, curr_step)
                    
                specific_coef_p = optimizer_set.get_specific_coef(
                    specialist_idx=specialist_idx)
                p_count = 0
                norm_sum = 0.
                for p in specific_coef_p:
                    if p.grad is not None:
                        norm_sum += torch.sum(torch.abs(p.grad.data))
                        p_count += np.array(p.grad.shape).sum()

                if p_count > 0:
                    specific_coef_p_avg_grad_norm = norm_sum / p_count
                else:
                    specific_coef_p_avg_grad_norm = 0
                
                writer.add_scalar('param_norm_specific_coefs',
                                  specific_coef_p_avg_grad_norm, curr_step)

            if log and batch_idx % config['log_interval'] == 0:
                print("{:.3f}%".format(100. * batch_idx / n_batches_per_epoch),
                        end=" ")
                sys.stdout.flush()

                print('Train Epoch: {} [{}/{} ({:.0f}%)]'.format(
                    epoch, batch_idx * len(x), n_samples,
                    100. * batch_idx / n_batches_per_epoch),
                    file=log_file)
                log_file.flush()


def test(config,
         log_file,
         test_loader,
         model,
         device,
         epoch,
         writer=None,
         log=True,
         dataset_mode='test',
         specialist=None,
         compute_nll=False):

    correct = 0.
    total = 0.
    loss = 0.

    with torch.no_grad():
        for batch_idx, (x, true_class) in enumerate(test_loader):
            if batch_idx == config['break_batch_idx']:
                break

            x = x.to(device)
            true_class = true_class.to(device)

            # Make a prediction.
            if specialist is not None:
                y = model(x, specialist=specialist)
            else:
                y = model(x)
            
            _, prediction = torch.max(y, 1)
            # Check whether prediciton is correct and accumulate.
            correct += int((prediction == true_class).sum().item())
            if compute_nll:
                if model.softmax_output():
                    loss += F.nll_loss(torch.log(y), true_class,
                                       reduction='sum')
                else:
                        loss += F.cross_entropy(y, true_class,
                                                reduction='sum')
                
            total += x.data.size()[0]

        if log:
            print('\n[Epoch {}, PID {}, run-id {}] {} set accuracy: {:.4f}'.format(
                epoch, str(os.getpid()), config['run_id'], dataset_mode, correct / total))
            print('\n[Epoch {}] {} set accuracy: {:.4f}'.format(
                epoch, dataset_mode, correct / total), file=log_file)
            log_file.flush()

        if writer is not None:
            writer.add_scalar(
                '{}_accuracy'.format(dataset_mode), correct/total,
                 epoch)
            if compute_nll:
                writer.add_scalar(
                    '{}_loss'.format(dataset_mode), loss,
                    epoch)
                print('\n[Epoch {}, PID {}, run-id {}] {} set nll: {:.4f}'.format(
                    epoch, str(os.getpid()), config['run_id'], dataset_mode, loss.item()))

    config['test_loss_dict'][epoch] = loss.item()
    return correct / total

def measure_entropy(config,
                    test_loader,
                    ensemble,
                    device,
                    return_list=False,
                    specialist=None):

    entropy_list = []
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(test_loader):
            if batch_idx == config['break_batch_idx']:
               break
            
            x = x.to(device)

            if return_list:
                if specialist is not None:
                    h = ensemble.entropy(x, specialist=specialist)
                else:
                    h = ensemble.entropy(x)
                h = h.detach().cpu()
            else:    
                if specialist is not None:
                    h = torch.mean(ensemble.entropy(x, specialist=specialist))
                else:
                    h = torch.mean(ensemble.entropy(x))
                h = h.detach().cpu().numpy().item()
            entropy_list += [h]

    if return_list:
        return entropy_list
    else:
        return np.mean(entropy_list)


def ood_metrics(in_dis, out_dis):
    assert(len(in_dis.shape) == len(out_dis.shape) == 1)
    with torch.no_grad():
        y_true = np.concatenate([np.zeros(in_dis.shape[0]),
                                                    np.ones(out_dis.shape[0])]).reshape(-1)
        y_scores = np.concatenate([in_dis, out_dis]).reshape(-1)
        return roc_auc_score(y_true, y_scores)

def test_ood(config,
             log_file,
             in_data_loader,
             out_data_loaders,
             ensemble,
             device,
             epoch,
             to_print=True,
             writer=None,
             specialist=None):

    
    in_dis_entropy_list = measure_entropy(config,
                                          in_data_loader,
                                          ensemble,
                                          device,
                                          return_list=True,
                                          specialist=specialist)

    in_dis_entropy_list = torch.cat(in_dis_entropy_list).numpy()
    
    if writer:
        writer.add_histogram('in_dis_dataset',
                             in_dis_entropy_list,
                             epoch)

    aurocs = []
    for i, out_data_loader in enumerate(out_data_loaders):
        out_dis_entropy_lists = measure_entropy(config,
                                                out_data_loader,
                                                ensemble,
                                                device,
                                                return_list=True,
                                                specialist=specialist)
    
        out_dis_entropy_lists = torch.cat(out_dis_entropy_lists).numpy()

        auroc = ood_metrics(in_dis_entropy_list, out_dis_entropy_lists)

        if log_file:
            print('\n[ood dataset {}, epoch {}] ood auroc: {:.4f}'
                .format(i, epoch, auroc), file=log_file)
            log_file.flush()
        
        if to_print:
            print('\n[ood dataset {}, epoch {}] ood auroc: {:.4f}'
                .format(config["ood_dataset_names"][i], epoch, auroc))

        aurocs.append(auroc)
        
        if writer:
            writer.add_histogram('ood_dataset_{}'.format(i), out_dis_entropy_lists, epoch)

            writer.add_scalar(
                    'ood_dataset_{}_ood_auroc'.format(config["ood_dataset_names"][i]), auroc, epoch)

    return aurocs
