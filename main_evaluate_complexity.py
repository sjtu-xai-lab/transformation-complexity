import os
import os.path as osp
import argparse
from tqdm import tqdm
import torch
import numpy as np

import models
from datasets import load_dataset
from metrics.calc_info import calc_information_for_epoch_KDE
from tools.utils import sample_input
from tools.lib import save_obj


def parse_args():
    parser = argparse.ArgumentParser("Evaluate Transformation Complexity")

    parser.add_argument('--gpu-id', type=int, default=0)

    parser.add_argument("--data-root", type=str, default="./data",
                        help="the root folder of datasets")
    parser.add_argument("--dataset", type=str, default="mnist",
                        help="the name of the dataset")
    parser.add_argument('--arch', type=str, default='mlp_mnist',
                        help='model architecture')
    parser.add_argument('--model-root', type=str, default='./saved-models/dataset=mnist_model=mlp',
                        help='the folder where models are saved')
    parser.add_argument('--epochs', type=int, default=501)
    parser.add_argument('--eval-interval', type=int, default=5)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    train_loader_sample, _ = load_dataset(dataset=args.dataset, data_root=args.data_root,
                                          batch_size=1, shuffle_train=False, data_aug_train=False)

    sampled_inputs, sampled_labels = sample_input(
        num_classes=10, data_loader=train_loader_sample, num_per_class=200, device=args.gpu_id
    )

    net = models.__dict__[args.arch]().to(args.gpu_id)

    epoch_list = list(range(0, args.epochs, args.eval_interval))
    HS_list = []
    ISY_list = []

    for e in tqdm(epoch_list):

        net.load_state_dict(torch.load(osp.join(args.model_root, f'model_{e}.pkl'),
                                       map_location=torch.device(f"cuda:{args.gpu_id}")))
        net.eval()

        with torch.no_grad():
            _ = net(sampled_inputs)
            sigma = [t.flatten(start_dim=1) for t in net.sigma_list]
            sigma = torch.cat(sigma, dim=1)
            for t in net.sigma_list:
                if len(t.shape) == 4:
                    sigma = sigma.unsqueeze(2).unsqueeze(3)
                    break

        sigma = [sigma, sampled_labels]
        network_info = calc_information_for_epoch_KDE(sigma, device=args.gpu_id)
        HS = network_info[0]['local_IXT']
        HS_list.append(HS)
        ISY = network_info[0]['local_ITY']
        ISY_list.append(ISY)

    save_obj({
        "epoch_list": epoch_list,
        "HS_list": HS_list,
        "ISY_list": ISY_list
    }, osp.join(args.model_root, "info_data.bin"))


if __name__ == '__main__':
    main()