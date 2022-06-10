import argparse
import os
import os.path as osp
import time
import json
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.optim

import models
import ebms
from ebms import ActRateTracker
from datasets import load_dataset
from tools.penalize_transformation import validate, train
from tools.lib import update_lr, save_obj, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--data-root", type=str, default="/data1/limingjie/data",
                        help="the root folder of datasets")
    parser.add_argument("--dataset", type=str, default="cifar10",
                        help="the name of the dataset")
    parser.add_argument('--arch', type=str, default='resmlp10',
                        help='model architecture')

    parser.add_argument('--model-lr', default=1e-2, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--model-lr-decay', default=2, type=float)
    parser.add_argument('--epochs', default=201, type=int, metavar='EPOCHS',
                        help='number of total epochs to run')
    parser.add_argument('--batch-size', default=128, type=int,
                        metavar='BS', help='mini-batch size (default: 128)')
    parser.add_argument('--default_train', action="store_true",
                        help='train without complexity loss')

    parser.add_argument("--penalize-layers", type=str, default="layers.5.act,layers.6.act,layers.7.act,layers.8.act",
                        help="which layers to penalize the transformation complexity")
    parser.add_argument("--energy-functions", type=str, default="E_3072d,E_3072d,E_3072d,E_3072d",
                        help="configuration of the energy function of each layer")
    parser.add_argument("--n-channels", type=str, default="3072,3072,3072,3072",
                        help="# of channels in gating layers")

    parser.add_argument('--ebm-lr', default=1e-4, type=float)
    parser.add_argument('--ebm-lr-decay', default=2, type=float)
    parser.add_argument('--l_step', default=5, type=int, metavar='L_STEP',
                        help='langevin step number')
    parser.add_argument('--tau', default=0.001, type=float, metavar='TAU',
                        help='tau is the step length in langevin step')
    parser.add_argument('--loss_lambda', default=-4.0, type=float, metavar='LAMBDA',
                        help='control the ratio between criterion loss and f function loss')
    parser.add_argument("--train-interval", type=int, default=10)

    parser.add_argument('--plot', action="store_true",
                        help='whether we should plot the loss function and train/test accuracy')
    parser.add_argument('--print-freq', '-p', default=50, type=int,
                        metavar='N', help='print every x batches (default: 50)')
    parser.add_argument("--save-folder", default="./saved-models")

    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)

    args = parser.parse_args()
    print(args)
    return args


def load_model(args):
    model = models.__dict__[args.arch](activation_type="swish").to(args.gpu_id)
    model.set_store_soft_gating_states(save_layers=args.penalize_layers)
    optimizer = torch.optim.SGD(model.parameters(), args.model_lr, momentum=0.9, weight_decay=5e-4)

    f_list = [ebms.__dict__[e_func]().to(args.gpu_id) for e_func in args.energy_functions.split(",")]
    q_list = [ActRateTracker(n_channels=int(n_c)).to(args.gpu_id) for n_c in args.n_channels.split(",")]
    f_optimizer_list = [
        torch.optim.SGD(f_list[i].parameters(), args.ebm_lr, momentum=0.9, weight_decay=5e-4)
        for i in range(len(f_list))
    ]
    return model, f_list, q_list, optimizer, f_optimizer_list


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


def main():
    args = parse_args()

    seed = args.seed
    set_seed(seed)

    if not args.default_train:
        args.save_folder = osp.join(args.save_folder, f"dataset={args.dataset}_model={args.arch}",
                                    f"M__bs={args.batch_size}_model-lr={args.model_lr}_epochs={args.epochs}_"
                                    f"E__l={args.l_step}_tau={args.tau}_lambda={args.loss_lambda}"
                                    f"_interval={args.train_interval}_seed={args.seed}")
    else:
        args.save_folder = osp.join(args.save_folder, f"dataset={args.dataset}_model={args.arch}",
                                    f"M__bs={args.batch_size}_model-lr={args.model_lr}"
                                    f"_epochs={args.epochs}_seed={args.seed}")
    os.makedirs(args.save_folder, exist_ok=True)
    os.makedirs(osp.join(args.save_folder, "model"), exist_ok=True)
    os.makedirs(osp.join(args.save_folder, "curve"), exist_ok=True)

    with open(osp.join(args.save_folder, "hparams.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    plot_dic = {
        "loss_label": [],
        "loss_compl": [],
        "loss_total": [],
        **{f"loss_compl_{i}": [] for i in range(len(args.penalize_layers.split(",")))},
        **{f"loss_f_{i}": [] for i in range(len(args.penalize_layers.split(",")))},
        **{f"p_{i}": [] for i in range(len(args.penalize_layers.split(",")))},
        "train_acc": [],
        "test_acc": []
    }

    model_lr_list = np.logspace(np.log10(args.model_lr), np.log10(args.model_lr) - args.model_lr_decay, args.epochs)
    f_lr_list = np.logspace(np.log10(args.ebm_lr), np.log10(args.ebm_lr) - args.ebm_lr_decay, args.epochs)

    train_loader, val_loader = load_dataset(dataset=args.dataset, data_root=args.data_root, batch_size=args.batch_size)
    model, f_list, q_list, optimizer, f_optimizer_list = load_model(args=args)
    assert len(f_list) == len(q_list)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        # update learning rates
        update_lr(optimizer, model_lr_list[epoch])
        for ebm_id in range(len(f_list)):
            update_lr(f_optimizer_list[ebm_id], f_lr_list[epoch])
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))

        # train and evaluate the model
        train(args, train_loader, model, f_list, q_list, criterion, optimizer, f_optimizer_list, epoch, plot_dic)
        prec1 = validate(val_loader, model, criterion, print_freq=args.print_freq, device=args.gpu_id)

        plot_dic["test_acc"].append(prec1)

        if args.plot or True:
            plot_curve(plot_dict=plot_dic, save_path=osp.join(args.save_folder, "curve", "curve.png"))

        save_obj(plot_dic, osp.join(args.save_folder, "curve", "data.bin"))

    torch.save(model.cpu().state_dict(), os.path.join(args.save_folder, "model", "model.pth"))
    for ebm_id in range(len(f_list)):
        torch.save(f_list[ebm_id].cpu().state_dict(), os.path.join(args.save_folder, "model", f"f{ebm_id}.pth"))
        torch.save(q_list[ebm_id].cpu().state_dict(), os.path.join(args.save_folder, "model", f"q{ebm_id}.pth"))

    save_stats(plot_dict=plot_dic, save_path=osp.join(args.save_folder, "model", "stats.txt"))


if __name__ == '__main__':
    main()