"""
Entry point for the SSL bearing fault diagnosis experiments.

Parses CLI args, builds the dataset/model/optimizer for the chosen
(--dataset, --method, --model) combination, runs the training loop for the
selected SSL method, then evaluates the best checkpoint and (for methods
with per-dataset t-SNE support) saves a feature visualization. The whole
run is repeated --ex-epochs times (with the seed incremented each time) and
results are aggregated into a CSV under ./result.
"""

import numpy as np
import pandas as pd
import argparse
import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils import *
from data_loader import *
from models import *
from train import *
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

parser = argparse.ArgumentParser(description="SSL_Bearing")

# Runtime / hardware
parser.add_argument('--gpu-id', default='0', type=int)
parser.add_argument('--local-rank', type=int, default=-1)  # set by torch.distributed launchers; -1 = single GPU, no DDP
parser.add_argument('--num-workers', default=0, type=int)  # DataLoader worker processes
parser.add_argument('--no-progress', action='store_true')

# Experiment selection
parser.add_argument('--dataset', default='hust', type=str,
                    choices=['slra', 'cwru', 'hust', 'pu'])
parser.add_argument('--method', default='proposed', type=str,
                    choices=['supervised', 'pseudo', 'hcae', 'mixmatch', 'fixmatch', 'simmatch', 'proposed'])
parser.add_argument('--model', default='cnn', type=str, choices=['cnn', 'wrn'])
parser.add_argument('--ex-epochs', default=10, type=int)  # number of repeated runs (seed incremented each time)
parser.add_argument('--epochs', default=1000, type=int)
parser.add_argument('--train-iteration', default=24, type=int)  # steps per epoch

# Ablation switches for --method proposed only (default off = original
# proposed method, unchanged). See ablation table:
#   neither flag        -> (4) proposed (masking aug + labeled mixup)
#   --no-mixup           -> (2) masking aug only
#   --no-mask-aug         -> (3) labeled mixup only
#   --no-mask-aug --no-mixup -> (1) no augmentation at all
parser.add_argument('--no-mask-aug', action='store_true')  # disables SpecTimeMask/SpecAugmentMask + the noise applied alongside them
parser.add_argument('--no-mixup', action='store_true')     # disables mixup_data_within_cls on labeled data
parser.add_argument('--legacy-mixup', action='store_true')  # uses mixup_data_within_cls_legacy (pre-fix: shared lam, no identity-permutation rejection) instead; for isolating that fix's effect. No-op if --no-mixup is also set.

# Loss-term ablation switches for --method proposed only (default off =
# original proposed method, unchanged). proposed's loss is
# labeled_loss + triplet_loss + entropy_u; see ablation table:
#   neither flag              -> CE + triplet + entropy (proposed, original)
#   --no-triplet                -> CE + entropy
#   --no-entropy                -> CE + triplet
#   --no-triplet --no-entropy -> CE only
parser.add_argument('--no-triplet', action='store_true')  # disables batch_triplet_loss (also skips computing it)
parser.add_argument('--no-entropy', action='store_true')  # disables the entropy penalty on unlabeled predictions

# Ablation: number of independently-masked views generated per labeled AND
# per unlabeled sample in --method proposed (Transform_Proposed_Multi /
# Transform_Proposed_ulb). Default 2 = original proposed method. Raising
# this increases the effective batch size (and memory/compute) per step by
# the same factor, since all views get concatenated into one forward pass.
parser.add_argument('--n-aug', default=2, type=int)

# Optimization
parser.add_argument('--lr', default=0.03, type=float)
parser.add_argument('--wdecay', default=5e-4, type=float)
parser.add_argument('--ema-decay', default=0.999, type=float)
parser.add_argument('--nesterov', action='store_true', default=True)
parser.add_argument('--use-ema', action='store_true', default=True)

# Data
parser.add_argument('--num-labeled', type=int, default=10)
parser.add_argument('--train-batch-size', default=8, type=int)
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--each-data-num', type=int, default=6)  # SLRA-only: files sampled per bearing type (class balancing)

# pseudo / hcae (shared ramp-up schedule for the unlabeled loss weight)
parser.add_argument('--T1', type=int, default=10)
parser.add_argument('--T2', type=int, default=100)
parser.add_argument('--max-alpha', type=int, default=1)

# mixmatch
parser.add_argument('--T', type=float, default=0.5)
parser.add_argument('--alpha', type=float, default=0.75)

# fixmatch (also reused by proposed/simmatch)
parser.add_argument('--threshold', type=float, default=0.95)
parser.add_argument('--lambda-u', type=int, default=1)

# proposed (batch_triplet_loss hinge margin, applied to L2-normalized features -- see utils.batch_triplet_loss)
parser.add_argument('--margin', type=float, default=0.5)
parser.add_argument('--legacy-margin', action='store_true')  # uses batch_triplet_loss_legacy (pre-fix: raw, non-normalized feature distances) instead; for isolating that fix's effect

parser.add_argument('--patience-limit', '--pl', default=20, type=int)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--tag', default='', type=str)  # appended to checkpoint/result/plot filenames (see utils.run_id); keeps concurrent runs (e.g. ablations) from overwriting each other's output

args = parser.parse_args()

def main():
    """
    Runs args.ex_epochs repeated experiments for the (args.dataset,
    args.method, args.model) combination: builds the data splits and model,
    trains with the selected SSL method, evaluates the best checkpoint on
    the test set, saves a t-SNE plot, and appends the run's accuracy/F1 to
    Test_acc. Results are written to a CSV once all repeats finish.
    """
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

    Test_acc = np.zeros((args.ex_epochs, 2))  # columns: [accuracy, macro-F1]

    # Per-dataset config: data folder, class count, val/test split size, and
    # SpecAugment mask widths for labeled/unlabeled augmentation
    if args.dataset == 'slra':
        data_path = './STFT_64'
        args.num_classes = 7
        args.val_num = 1000
        args.lab_mask = 10
        args.unlab_mask = 5
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

        train_loader = DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True, drop_last=True,
                                  num_workers=args.num_workers)
        val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers)
        test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers)
        if args.method != 'supervised':
            unlabeled_loader = DataLoader(unlabelset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                          num_workers=args.num_workers)

        if args.local_rank == 0:
            torch.distributed.barrier()

        model = create_model(args)

        if args.local_rank == 0:
            torch.distributed.barrier()

        # Skip weight decay on bias/BatchNorm parameters (standard practice)
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

        # Note: each *_train() function's `testloader` parameter is passed
        # val_loader here — it's used for early stopping / checkpoint
        # selection during training. The held-out test_loader is only
        # touched afterward, once, for the final reported metrics below.
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
            torch.load(f'./result/{run_id(args)}.pth', weights_only=True))
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
    Test_acc.to_csv(f'./result/{run_id(args)}_result.csv', index=False)

    print('end')


if __name__ == '__main__':
    main()