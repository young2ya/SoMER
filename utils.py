"""
Utility functions for training and evaluation.

This module containes:
    - Reproducibility helpers (set seed)
    - Test/evaluation helpers (test)
    - t-SNE visualization utilities per dataset
    - SSL helpers: interleaving, MixMatch loss weighting, similarity graph utilities
    - Proposed method utilities: feature memory, entropy, soft triplet loss, class-wise mixup
"""

import os
import numpy as np
import random
import time
from tqdm import tqdm

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import LambdaLR

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark= False
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1,-1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res

def test(args, loader, model):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()
    end = time.time()
    
    all_preds = []
    all_targets = []

    if not args.no_progress:
        loader = tqdm(loader, disable=args.local_rank not in [-1, 0])
        
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            model.eval()
            
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            _, outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            
            prec1, prec2 = accuracy(outputs, targets, topk=(1,2))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top2.update(prec2.item(), inputs.shape[0])

            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                loader.set_description(                    
                    "Test Iter: {batch:4}/{iter:4}. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top2: {top2:.2f}. ".format(
                        batch=batch_idx + 1,
                        iter=len(loader),
                        bt=batch_time.avg,
                        loss=losses.avg,
                        top1=top1.avg,
                        top2=top2.avg,
                    ))
        if not args.no_progress:
            loader.close()

        macro_f1 = f1_score(all_targets, all_preds, average='macro')
            
    print('Accuracy : {:.4f} | Macro-F1 : {:.4f}'.format(top1.avg, macro_f1))
    return losses.avg, top1.avg, macro_f1

def bearing_tsne(args, test_loader, model, repeat):
    features = []
    targets_all = []
    
    if not args.no_progress:
        test_loader = tqdm(test_loader, disable=args.local_rank not in [-1,0])
        
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            model.eval()
            
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            _, outputs = model(inputs)
            
            features.append(outputs.cpu().numpy())
            targets_all.append(targets.cpu().numpy())
            
        features = np.concatenate(features)
        targets_all = np.concatenate(targets_all)
        
        tsne = TSNE(n_components=2, random_state=args.seed)
        tsne_results = tsne.fit_transform(features)
        
        plt.figure(figsize=(10,10))
        bearing_classes = ['N', 'SB', 'SI', 'SO', 'WB', 'WI', 'WO']
        colors = plt.cm.tab10(np.linspace(0,1,len(bearing_classes)))
        
        for i,class_name in enumerate(bearing_classes):
            indices = targets_all == i
            plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1],
                        color= colors[i], label=class_name, alpha=0.7)
            
        plt.legend(loc='best')
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimention 2")
        plt.title(f"t-SNE of {args.method}")
        plt.savefig(f"./plot/{args.model}_{args.method}_{args.num_labeled}_{repeat}.png")
        plt.close()
    return

def cwru_tsne(args, test_loader, model, repeat):
    features = []
    targets_all = []

    if not args.no_progress:
        test_loader = tqdm(test_loader, disable=args.local_rank not in [-1,0])

    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            _, outputs = model(inputs)

            features.append(outputs.cpu().numpy())
            targets_all.append(targets.cpu().numpy())

    features = np.concatenate(features)
    targets_all = np.concatenate(targets_all)

    tsne = TSNE(n_components= 2, random_state=args.seed)
    tsne_results = tsne.fit_transform(features)

    cwru_classes = ['N', 'B007', 'B014', 'B021', 'I007', 'I014', 'I021', 'O007', 'O014', 'I021']
    colors = plt.cm.tab10(np.linspace(0, 1, len(cwru_classes)))

    plt.figure(figsize=(10,10))
    for i, class_name in enumerate(cwru_classes):
        indices = targets_all == i
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1],
        color=colors[i], label=class_name, alpha=0.7)

    plt.legend(loc='best')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title(f"t-SNE of {args.method} on CWRU")
    os.makedirs('./plot', exist_ok=True)
    plt.savefig(f"./plot/{args.model}_{args.method}_{args.num_labeled}_{repeat}_CWRU.png")
    plt.close()

def hust_tsne(args, test_loader, model, repeat):
    features = []
    targets_all = []

    if not args.no_progress:
        test_loader = tqdm(test_loader, disable=args.local_rank not in [-1,0])

    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            _, outputs = model(inputs)

            features.append(outputs.cpu().numpy())
            targets_all.append(targets.cpu().numpy())

    features = np.concatenate(features)
    targets_all = np.concatenate(targets_all)

    tsne = TSNE(n_components= 2, random_state=args.seed)
    tsne_results = tsne.fit_transform(features)

    hust_classes = ['N', 'I', 'O', 'B', 'IO', 'IB', 'OB']
    colors = plt.cm.tab10(np.linspace(0, 1, len(hust_classes)))

    plt.figure(figsize=(10,10))
    for i, class_name in enumerate(hust_classes):
        indices = targets_all == i
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1],
        color=colors[i], label=class_name, alpha=0.7)

    plt.legend(loc='best')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title(f"t-SNE of {args.method} on HUST")
    os.makedirs('./plot', exist_ok=True)
    plt.savefig(f"./plot/{args.model}_{args.method}_{args.num_labeled}_{repeat}_HUST.png")
    plt.close()

def pu_tsne(args, test_loader, model, repeat):
    features = []
    targets_all = []

    if not args.no_progress:
        test_loader = tqdm(test_loader, disable=args.local_rank not in [-1,0])

    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            _, outputs = model(inputs)

            features.append(outputs.cpu().numpy())
            targets_all.append(targets.cpu().numpy())

    features = np.concatenate(features)
    targets_all = np.concatenate(targets_all)

    tsne = TSNE(n_components= 2, random_state=args.seed)
    tsne_results = tsne.fit_transform(features)

    pu_classes = ['N', 'I', 'O']
    colors = plt.cm.tab10(np.linspace(0, 1, len(pu_classes)))

    plt.figure(figsize=(10,10))
    for i, class_name in enumerate(pu_classes):
        indices = targets_all == i
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1],
        color=colors[i], label=class_name, alpha=0.7)

    plt.legend(loc='best')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title(f"t-SNE of {args.method} on PU")
    os.makedirs('./plot', exist_ok=True)
    plt.savefig(f"./plot/{args.model}_{args.method}_{args.num_labeled}_{repeat}_PU.png")
    plt.close()

# FixMatch
def fix_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0,1).reshape([-1] + s[1:])

def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0,1).reshape([-1] + s[1:])

# MixMatch
def interleave_offset(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets

def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offset(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]

def linear_ranmpup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)
    
class MixLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, args):
        probs_u = torch.softmax(outputs_u, dim=1)
        
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)
        
        return Lx, Lu, args.lambda_u * linear_ranmpup(epoch, args.epochs)
    
# SimMatch
from collections import deque
def _normalize(feat: torch.Tensor) -> torch.Tensor:
    """L2-normalize feature vectors along the channel dimension"""
    return F.normalize(feat, dim=1)

def build_similarity(feat: torch.Tensor) -> torch.Tensor:
    """Cosine-similarity matrix (N x N) for a mini-batch"""
    feat_n = _normalize(feat)
    return torch.mm(feat_n, feat_n.t())

def build_semantic_graph(labels: torch.Tensor) -> torch.Tensor:
    """Binary dajacency: 1 if same pseudo-label, else 0 (N x N)"""
    return (labels.unsqueeze(1) == labels.unsqueeze(0)).float()

def mse_detach(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE where the first argument is detached (stop-grad)"""
    return F.mse_loss(source.detach(), target)

class FeatureMemory:
    """Circular buffer holding (features, labels) for graph refinement"""
    def __init__(self, capacity: int = 4096, device: str = 'cpu'):
        self.capacity = capacity
        self.device = device
        self.buffer = deque(maxlen=capacity)
        
    def enqueue(self, feat: torch.Tensor, label: torch.Tensor):
        feat = feat.detach().to(self.device)
        label = label.detach().to(self.device)
        for f, l in zip(feat, label):
            self.buffer.append((f,l))
            
    def get(self):
        if len(self.buffer) == 0:
            return None, None
        feats, labels = zip(*self.buffer)
        return torch.stack(feats, dim=0), torch.stack(labels, dim=0)

# Proposed
def calculate_entropy(probs):
    eps = 1e-8
    entropy = -probs * torch.log(probs + eps)
    return entropy.sum(dim=1)

def calculate_euclidean_distance(single_tensor, batch_tensor):
    # reshape single_tensor
    single_tensor_expanded = single_tensor.unsqueeze(0)
    
    # calculate the squared differences
    diff = batch_tensor - single_tensor_expanded
    squared_diff = diff ** 2
    
    # sum over the dimensions to get the squared distances
    squared_distances = torch.sum(squared_diff, dim=1)
    
    # take the square root to get the euclidean distances
    distances = torch.sqrt(squared_distances)
    return distances

def batch_triplet_loss(args, inputs, targets, all_inputs, class_probs):
    losses = []
    margin = 0.5
    
    for anchor_data, anchor_label in zip(inputs, targets):
        positive_probs = class_probs[:, anchor_label]
        
        other_classes = [i for i in range(args.num_classes) if i != anchor_label]
        
        # negative_probs, _ = class_probs[:, other_classes].max(dim=1)

        #appendix
        negative_probs = 1.0 - positive_probs

        distance = calculate_euclidean_distance(anchor_data, all_inputs)
        
        pos_dist = positive_probs * distance
        neg_dist = negative_probs * distance
        
        batch_loss = torch.clamp(pos_dist - neg_dist + margin, min=0.0)
        
        losses.append(batch_loss.mean())
    return torch.stack(losses).mean()

def mixup_data_within_cls(x, y, alpha=1.0, n_mixups=2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
        
    mixed_x = torch.zeros_like(x)
    unique_classes = y.unique()
    mixed_data_list = []
    mixed_labels_list = []
    
    for cls in unique_classes:
        indices = (y == cls).nonzero(as_tuple=True)[0]
        if len(indices) > 1:
            for _ in range(n_mixups):
                shuffled_indices = indices[torch.randperm(len(indices))]
                mixed_x[indices] = lam * x[indices] + (1 - lam) * x[shuffled_indices]
                
                mixed_data_list.append(mixed_x[indices])
                mixed_labels_list.append(y[indices])
                
            mixed_data_list.append(x[indices])
            mixed_labels_list.append(y[indices])
        else:
            mixed_data_list.append(x[indices])
            mixed_labels_list.append(y[indices])
            
    mixed_x = torch.cat(mixed_data_list, dim=0)
    mixed_y = torch.cat(mixed_labels_list, dim=0)
    return mixed_x, mixed_y