import argparse
import os
import os.path as osp
import time
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim

import models
import ebms
from ebms import ActRateTracker
from datasets import load_dataset, get_num_classes
from tools.penalize_transformation import validate, train, plot_curve, save_stats, sample_input, eval_acc_loss
from metrics.calc_info import calc_information_for_epoch_KDE
from tools.lib import update_lr, save_obj, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Penalize the transformation complexity')
    parser.add_argument("--data-root", type=str, default="./data",
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

    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--num-per-class", type=int, default=200)

    args = parser.parse_args()
    print()
    print(args)
    print()
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


def setup(args):
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

    if not args.evaluate:
        os.makedirs(args.save_folder, exist_ok=True)
        os.makedirs(osp.join(args.save_folder, "model"), exist_ok=True)
        os.makedirs(osp.join(args.save_folder, "curve"), exist_ok=True)

        with open(osp.join(args.save_folder, "hparams.json"), "w") as f:
            json.dump(vars(args), f, indent=4)
    else:
        args.model_path = osp.join(args.save_folder, "model", "model.pth")
        os.makedirs(osp.join(args.save_folder, "evaluate"), exist_ok=True)

        with open(osp.join(args.save_folder, "hparams_eval.json"), "w") as f:
            json.dump(vars(args), f, indent=4)


def main(args):

    setup(args)

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


def main_evaluate(args):

    setup(args)

    # load dataset and model
    train_loader, val_loader = load_dataset(dataset=args.dataset, data_root=args.data_root,
                                            batch_size=args.batch_size, shuffle_train=False, data_aug_train=False)
    train_loader_sample, _ = load_dataset(dataset=args.dataset, data_root=args.data_root,
                                          batch_size=1, shuffle_train=False, data_aug_train=False)
    model, _, _, _, _ = load_model(args=args)
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device(f"cuda:{args.gpu_id}")))
    model.eval()
    print(f"Model loaded from {args.model_path}")

    # sample some input images for calculating transformation complexity (See Appendix G.1)
    num_classes = get_num_classes(args.dataset)
    input, label = sample_input(num_classes=num_classes, num_per_class=args.num_per_class,
                                data_loader=train_loader_sample, device=args.gpu_id)

    # generating the gating states
    with torch.no_grad():
        model(input)
    num_samples = input.shape[0]
    sigma = [model.sigma_list[layer_id].reshape(num_samples, -1) for layer_id in model.sigma_list.keys()]
    sigma = torch.cat(sigma, dim=1)
    sigma_label = [sigma, label]

    # calculate transformation complexity
    ret_dict = {}
    network_info = calc_information_for_epoch_KDE(sigma_label, device=args.gpu_id)
    HS = network_info[0]['local_IXT']  # H(\Sigma)
    ISY = network_info[0]['local_ITY']  # I(X;\Sigma;Y)
    ret_dict["HS"] = HS.item()
    ret_dict["ISY"] = ISY.item()

    # calculate loss and accuracy
    _, train_loss = eval_acc_loss(model, train_loader, device=args.gpu_id)
    test_acc, test_loss = eval_acc_loss(model, val_loader, device=args.gpu_id)
    ret_dict["train_loss"] = train_loss
    ret_dict["test_loss"] = test_loss
    ret_dict["test_acc"] = test_acc

    save_obj(ret_dict, osp.join(args.save_folder, "evaluate", "info.bin"))
    with open(osp.join(args.save_folder, "evaluate", "info.json"), "w") as f:
        json.dump(ret_dict, f, indent=4)

    return ret_dict


if __name__ == '__main__':
    args = parse_args()
    if not args.evaluate:
        main(args)
    else:
        main_evaluate(args)
