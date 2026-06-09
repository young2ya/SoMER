import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from tqdm import tqdm
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from sklearn.model_selection import train_test_split

from utils import *
from data_loader import *
from models import *
from train import *
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

parser = argparse.ArgumentParser(description="SSL_Bearing")

parser.add_argument('--gpu-id', default='0', type=int)
parser.add_argument('--local-rank', type=int, default=-1)
parser.add_argument('--num-workers', default=0, type=int)
parser.add_argument('--no-progress', action='store_true')

parser.add_argument('--dataset', default='hust', type=str,
                    choices=['slra', 'cwru', 'hust', 'pu'])
parser.add_argument('--method', default='proposed', type=str,
                    choices=['supervised', 'pseudo', 'hcae', 'mixmatch', 'fixmatch', 'simmatch', 'proposed'])
parser.add_argument('--model', default='cnn', type=str, choices=['cnn', 'wrn'])
parser.add_argument('--ex-epochs', default=10, type=int)
parser.add_argument('--epochs', default=1000, type=int)
parser.add_argument('--train-iteration', default=16, type=int)

parser.add_argument('--lr', default=0.03, type=float)
parser.add_argument('--wdecay', default=5e-4, type=float)
parser.add_argument('--ema-decay', default=0.999, type=float)
parser.add_argument('--nesterov', action='store_true', default=True)
parser.add_argument('--use-ema', action='store_true', default=True)

parser.add_argument('--num-labeled', type=int, default=10)
parser.add_argument('--train-batch-size', default=8, type=int)
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--each-data-num', type=int, default=6)

# pseudo
parser.add_argument('--T1', type=int, default=10)
parser.add_argument('--T2', type=int, default=100)
parser.add_argument('--max-alpha', type=int, default=1)

# mixmatch
parser.add_argument('--T', type=float, default=0.5)
parser.add_argument('--alpha', type=float, default=0.75)
parser.add_argument('--lambda', type=float, default=75)

# fixmatch
parser.add_argument('--threshold', type=float, default=0.95)
parser.add_argument('--lambda-u', type=int, default=1)
parser.add_argument('--mu', default=2, type=int)

parser.add_argument('--lambda-st', default=2, type=int)
parser.add_argument('--patience-limit', '--pl', default=20, type=int)
parser.add_argument('--seed', default=0, type=int)

args = parser.parse_args()

def main():
    if args.local_rank == -1:
        device = torch.device('cuda', args.gpu_id)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1

    args.device = device

    Test_acc = np.zeros((args.ex_epochs, 2))
    if args.dataset == 'slra':
        data_path = './STFT_128'
        args.num_classes = 7
        args.val_num = 1000
        args.lab_mask = 20
        args.unlab_mask = 10
    elif args.dataset == 'cwru':
        data_path = './CWRU_64'
        args.num_classes = 10
        args.val_num = 200
        args.lab_mask = 10
        args.unlab_mask = 5
    elif args.dataset == 'hust':
        data_path = './HUST_64'
        args.num_classes = 7
        args.val_num = 200
        args.lab_mask = 10
        args.unlab_mask = 5
    elif args.dataset == 'pu':
        data_path = './PU_64'
        args.num_classes = 3
        args.val_num = 1000
        args.lab_mask = 10
        args.unlab_mask = 5
        
    path_list = [os.path.join(data_path, f_name) for f_name in os.listdir(data_path)]

    for i in range(args.ex_epochs):
        print(f"repeat experiment: {i + 1}")

        if args.seed is not None:
            set_seed(args)

        if args.method == 'supervised':
            trainset, valset, testset = get_data(args, path_list)
        else:
            trainset, unlabelset, valset, testset = get_data(args, path_list)

        train_loader = DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)
        if args.method != 'supervised':
            unlabeled_loader = DataLoader(unlabelset, batch_size=args.batch_size, shuffle=True, drop_last=True)

        if args.local_rank == 0:
            torch.distributed.barrier()

        model = create_model(args)

        if args.local_rank == 0:
            torch.distributed.barrier()

        no_decay = ['bias', 'bn']
        grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
            {'params': [p for n, p in model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        if args.use_ema:
            ema_model = ModelEMA(args, model, args.ema_decay)

        optimizer = optim.SGD(grouped_parameters, lr=args.lr, momentum=0.9, nesterov=args.nesterov)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        criterion = MixLoss()

        if args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.local_rank],
                output_device=args.local_rank, find_unused_parameters=True)

        patience_check = 0
        if args.method == 'supervised':
            supervised_train(args, train_loader, val_loader, model, optimizer, scheduler, ema_model, patience_check)
        elif args.method == 'pseudo':
            Pseudo_train(args, train_loader, unlabeled_loader, val_loader, model, optimizer, scheduler, ema_model,
                         patience_check)
        elif args.method == 'hcae':
            HCAE_train(args, train_loader, unlabeled_loader, val_loader, model, optimizer, scheduler, ema_model,
                       patience_check)
        elif args.method == 'mixmatch':
            MixMatch_train(args, train_loader, unlabeled_loader, val_loader, model, criterion, optimizer, scheduler,
                           ema_model, patience_check)
        elif args.method == 'fixmatch':
            FixMatch_train(args, train_loader, unlabeled_loader, val_loader, model, optimizer, scheduler, ema_model,
                           patience_check)
        elif args.method == 'simmatch':
            SimMatch_train(args, train_loader, unlabeled_loader, val_loader, model, optimizer, scheduler, ema_model,
                           patience_check)
        elif args.method == 'proposed':
            Proposed_train(args, train_loader, unlabeled_loader, val_loader, model, optimizer, scheduler, ema_model,
                           patience_check)

        model.load_state_dict(
            torch.load(f'./result/{args.model}_{args.method}_{args.num_labeled}_{args.dataset}.pth', weights_only=True))
        test_loss, test_acc, test_f1 = test(args, test_loader, model)

        Test_acc[i][0] = test_acc
        Test_acc[i][1] = test_f1
        print(f"Test acc: {test_acc} | Macro-F1: {test_f1}")

        if args.dataset == 'slra':
            bearing_tsne(args, test_loader, model, i)
        elif args.dataset == 'cwru':
            cwru_tsne(args, test_loader, model, i)
        elif args.dataset == 'hust':
            hust_tsne(args, test_loader, model, i)
        elif args.dataset == 'pu':
            pu_tsne(args, test_loader, model, i)

        if args.seed is not None:
            args.seed = args.seed + 1

    Test_acc = pd.DataFrame(Test_acc, columns=['Accuracy', 'Macro-F1'])
    Test_acc.to_csv(f'./result/{args.model}_{args.method}_{args.num_labeled}_{args.dataset}_result.csv', index=False)

    print('end')


if __name__ == '__main__':
    main()