import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib
from typing import Tuple
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .lib import AverageMeter


def accuracy(output, target, topk=(1,)):
    """
    Computes the precision@k for the specified values of k
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def train_f(f, q, f_optimizer, sigma, langevin_step_l, delta_tau, device):
    f.train()
    q.train()
    sigma_tilt, p_0 = langevin_dynamics(sigma, langevin_step_l, delta_tau, f, q, device)
    loss_f = torch.mean(f(sigma_tilt.detach()) - f(sigma.detach()))
    f_optimizer.zero_grad()
    loss_f.backward()
    nn.utils.clip_grad_value_(f.parameters(), 1e-5)
    f_optimizer.step()
    f.eval()
    q.eval()
    return loss_f


def train(args, train_loader, model, f_list, q_list, criterion, optimizer, f_optimizer_list, epoch, plot_dic):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    task_losses = AverageMeter()
    complexity_losses = AverageMeter()
    f_losses_list = [AverageMeter() for _ in range(len(f_list))]
    compl_losses_list = [AverageMeter() for _ in range(len(f_list))]
    top1 = AverageMeter()

    # parameters for the langevin dynamics
    delta_tau = args.tau
    langevin_step_l = args.l_step

    # switch to train mode
    model.train()
    end = time.time()

    for i, (input, target) in enumerate(tqdm(train_loader, desc="training", mininterval=1)):
        # measure data loading time
        data_time.update(time.time() - end)
        current_batch_size = input.size(0)

        input, target = input.to(args.gpu_id), target.to(args.gpu_id)
        output = model(input)

        task_loss = criterion(output, target)
        prec1 = accuracy(output.data, target)[0]

        if not args.default_train:
            # update the energy function (every iteration)
            loss_f_list = []
            for ebm_id in range(len(f_list)):
                loss_f = train_f(
                    f=f_list[ebm_id], q=q_list[ebm_id], f_optimizer=f_optimizer_list[ebm_id],
                    sigma=model.sigma_list[ebm_id], langevin_step_l=langevin_step_l,
                    delta_tau=delta_tau, device=args.gpu_id
                )
                loss_f_list.append(loss_f)

            # calculate complexity loss in Eq. (4)
            #  Note: Actually, the complexity loss is optimized every `train_interval` iterations.
            #        However, to better keep track of the complexity loss, it is calculated every iteration.
            #        Nevertheless, this code can be revised for better time efficiency.
            loss_compl_list = []
            for ebm_id in range(len(f_list)):
                loss_compl = torch.mean(energy_function(f_list[ebm_id], q_list[ebm_id], model.sigma_list[ebm_id]))
                if len(model.sigma_list[ebm_id].shape) > 2:
                    n_positions = np.prod(model.sigma_list[ebm_id].shape[2:])  # the avg loss at each position
                else:
                    n_positions = 1
                loss_compl_list.append(loss_compl / n_positions)
        else:
            loss_f_list = [torch.zeros([]) for _ in range(len(f_list))]
            loss_compl_list = [torch.zeros([]) for _ in range(len(f_list))]

        if not args.default_train:
            complexity_loss = sum(loss_compl_list) * (10 ** args.loss_lambda)
        else:
            complexity_loss = sum(loss_compl_list) * 0

        # update the model
        optimizer.zero_grad()
        if i > 0 and i % args.train_interval == 0:
            loss = task_loss + complexity_loss
            loss.backward()
        else:
            loss = task_loss
            loss.backward()
        optimizer.step()

        # store stats
        losses.update(task_loss.item() + complexity_loss.item(), current_batch_size)
        for ebm_id in range(len(f_list)):
            compl_losses_list[ebm_id].update(loss_compl_list[ebm_id].item(), current_batch_size)
            f_losses_list[ebm_id].update(loss_f_list[ebm_id].item(), current_batch_size)
        task_losses.update(task_loss.item(), current_batch_size)
        complexity_losses.update(complexity_loss.item(), current_batch_size)
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('\n\tEpoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    # for plot
    plot_dic["loss_total"].append(losses.avg)
    plot_dic["loss_label"].append(task_losses.avg)
    plot_dic["loss_compl"].append(complexity_losses.avg)

    for ebm_id in range(len(f_list)):
        plot_dic[f"loss_compl_{ebm_id}"].append(compl_losses_list[ebm_id].avg)
        plot_dic[f"loss_f_{ebm_id}"].append(f_losses_list[ebm_id].avg)
        plot_dic[f"p_{ebm_id}"].append(q_list[ebm_id].calculate_mean_act_rate())
    plot_dic["train_acc"].append(top1.avg)


def validate(val_loader, model, criterion, print_freq, device):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            # print(target)
            target = target.to(device)
            input_var = input.to(device)
            target_var = target

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg


def langevin_dynamics(sigma_l, step_l, delta_tau, f, q, device):
    global feature_count, feature_map_path
    f.eval()

    if len(sigma_l.shape) == 2:
        noise = torch.zeros(1, sigma_l.shape[1]).to(device)
    else:
        noise = torch.zeros(1, sigma_l.shape[1], sigma_l.shape[2], sigma_l.shape[3]).to(device)  # use different noise in different batches

    sigma_tilt = sigma_l.clone().detach()
    sigma_tilt.requires_grad = True
    for i in range(step_l):
        E = - f(sigma_tilt) - q(sigma_tilt)

        q.eval()
        ones = torch.ones(E.size()).to(device)
        E.backward(ones, retain_graph=True)
        noise_term = (np.sqrt(delta_tau) * noise.normal_(mean=0, std=1)).to(device)
        sigma_tilt.data = sigma_tilt.data - 0.5 * delta_tau * (sigma_tilt.grad.data) + noise_term
        sigma_tilt.grad.zero_()

    return sigma_tilt.detach(), q.p_hat.data


def energy_function(f, q, sigma):
    return - f(sigma) - q(sigma)


def plot_curve(plot_dict: dict, save_path: str):
    num_figures = len(plot_dict) - 1
    n_rows = int(np.ceil(num_figures / 4))

    plt.figure(figsize=(16, 3.75 * n_rows), dpi=200)

    epoch = len(plot_dict[list(plot_dict.keys())[0]])
    X = np.arange(1, epoch + 1)

    plt.subplot(n_rows, 4, 1)
    plt.xlabel('epoch')
    plt.ylabel(r'$Loss_{DNN}=Loss_{label}+\lambda Loss_{compl.}$')
    plt.plot(X, plot_dict["loss_total"], label=r"$Loss_{DNN}$")
    plt.legend()

    plt.subplot(n_rows, 4, 2)
    plt.xlabel('epoch')
    plt.ylabel('train_test_precision')
    plt.plot(X, plot_dict["train_acc"], color='b', label="train_acc")
    plt.plot(X, plot_dict["test_acc"], color='r', label="test_acc")
    plt.legend()

    plt.subplot(n_rows, 4, 3)
    plt.xlabel('epoch')
    plt.ylabel(r'$Loss_{task}$')
    plt.plot(X, plot_dict["loss_label"], label=r"$Loss_{task}$")
    plt.legend()

    plt.subplot(n_rows, 4, 4)
    plt.xlabel('epoch')
    plt.ylabel(r'\lambda $Loss_{compl.}$')
    plt.plot(X, plot_dict["loss_compl"], label=r"$\lambda Loss_{compl.}$")
    plt.legend()

    fig_id = 5
    ebm_id = 0
    while True:
        if f"loss_compl_{ebm_id}" not in plot_dict:
            break
        plt.subplot(n_rows, 4, fig_id)
        plt.xlabel('epoch')
        plt.ylabel(r'$Loss_{compl.,' + f'{ebm_id}' + r'}$')
        plt.plot(X, plot_dict[f"loss_compl_{ebm_id}"], label=r'$Loss_{compl.,' + f'{ebm_id}' + r'}$')
        plt.legend()
        ebm_id += 1
        fig_id += 1

    ebm_id = 0
    while True:
        if f"loss_f_{ebm_id}" not in plot_dict:
            break
        plt.subplot(n_rows, 4, fig_id)
        plt.xlabel('epoch')
        plt.ylabel(r'$L_{f, ' + f'{ebm_id}' + r'}$')
        plt.plot(X, plot_dict[f"loss_f_{ebm_id}"], label=r'$L_{f, ' + f'{ebm_id}' + r'}$')
        plt.legend()
        ebm_id += 1
        fig_id += 1

    ebm_id = 0
    while True:
        if f"p_{ebm_id}" not in plot_dict:
            break
        plt.subplot(n_rows, 4, fig_id)
        plt.xlabel('epoch')
        plt.ylabel(r"$\hat{p}_" + f"{ebm_id}" + r"$")
        plt.plot(X, plot_dict[f"p_{ebm_id}"], label=r"$\hat{p}_" + f"{ebm_id}" + r"$")
        plt.legend()
        ebm_id += 1
        fig_id += 1

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close("all")


def save_stats(plot_dict: dict, save_path: str):
    item_length = len(plot_dict[list(plot_dict.keys())[0]])
    for key in plot_dict.keys():
        assert len(plot_dict[key]) == item_length
    with open(save_path, "w") as f:
        print(",".join(list(plot_dict.keys())), file=f)
        for i in range(item_length):
            for key in plot_dict.keys():
                f.write(f"{plot_dict[key][i]},")
            f.write("\n")
    return


def eval_acc_loss(
        model: nn.Module,
        data_loader: DataLoader,
        device: int = None
):
    if device is None:
        device = torch.device("cpu")
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        losses = AverageMeter()
        correct = 0
        model.eval()
        with torch.no_grad():
            # for images, labels in tqdm(data_loader, desc="train", mininterval=1):
            for images, labels in data_loader:
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                loss = criterion(output, labels)
                y_pred = output.data.max(1)[1]
                correct += y_pred.eq(labels.data).sum()
                losses.update(loss.item(), images.shape[0])
        acc = 100. * float(correct) / len(data_loader.dataset)
    return acc, losses.avg
