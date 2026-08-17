"""
Training loops for each SSL method supported by main.py.

Every function follows the same skeleton:
    - iterate for args.epochs, each epoch running args.train_iteration steps
      (steps, not full passes over the loader — labeled/unlabeled iterators
      are re-created via try/except StopIteration when they run out)
    - compute a method-specific loss, backprop, step the optimizer/scheduler,
      and update the EMA shadow model if args.use_ema
    - evaluate on the val/test loader once per epoch, save a checkpoint when
      test loss improves, and early-stop after args.patience_limit epochs
      without improvement. A NaN test loss (e.g. from a diverged run) is
      never treated as an improvement, so it can't overwrite a good
      checkpoint with a corrupted one.
"""

import math
import time
from copy import deepcopy
import torch.nn as nn
import torch.nn.functional as F

from utils import *


def supervised_train(args, trainloader, testloader, model, optimizer, scheduler, ema_model, patience):
    """
    Fully-supervised baseline (--method supervised): trains only on the
    labeled set, using Transform_Proposed_Multi's multi-view augmentation
    plus within-class mixup for extra regularization on the few labeled
    samples.
    """
    test_accs = []
    end = time.time()
    best_loss = float('inf')

    if args.world_size > 1:
        labeled_epoch = 0
        trainloader.sampler.set_epoch(labeled_epoch)

    model.train()
    for epoch in range(args.epochs):
        batch_time = AverageMeter()
        losses = AverageMeter()

        if not args.no_progress:
            p_bar = tqdm(range(args.train_iteration), disable=args.local_rank not in [-1, 0])

        for batch_idx, (aug_list, targets) in enumerate(trainloader):
            inputs = torch.cat(aug_list, dim=0).to(args.device)
            targets = targets.long()
            targets = torch.cat([targets for _ in range(len(aug_list))], dim=0).to(args.device)
            
            if getattr(args, 'legacy_mixup', False):
                inputs, targets = mixup_data_within_cls_legacy(inputs, targets)
            else:
                inputs, targets = mixup_data_within_cls(inputs, targets)

            _, outputs = model(inputs)

            loss = F.cross_entropy(outputs, targets)
            losses.update(loss.item(), inputs.size(0))

            loss.backward()
            optimizer.step()
            scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()
            optimizer.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                p_bar.set_description(
                    "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}.  Batch: {bt:.3f}s. Loss: {loss:.4f}.".format(
                        epoch=epoch + 1,
                        epochs=args.epochs,
                        batch=batch_idx + 1,
                        iter=args.train_iteration,
                        lr=scheduler.get_last_lr()[0],
                        bt=batch_time.avg,
                        loss=losses.avg))
                p_bar.update()

        if not args.no_progress:
            p_bar.close()

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        if args.local_rank in [-1, 0]:
            test_loss, test_acc, test_f1 = test(args, testloader, test_model)

            test_accs.append(test_acc)

            # A NaN test loss (e.g. from a diverged run) must never look
            # like an improvement: Python/NumPy NaN comparisons are always
            # False, so a bare `test_loss >= best_loss` would silently fall
            # into the `else` branch below and save a corrupted checkpoint.
            if math.isnan(test_loss) or test_loss >= best_loss:
                patience += 1
                if patience >= args.patience_limit:
                    break
            else:
                best_loss = deepcopy(test_loss)
                patience = 0
                torch.save(model.state_dict(), f'./result/{run_id(args)}.pth')
            print(f"patience : {patience + 1}/{args.patience_limit}")

def Pseudo_train(args, labeled_loader, unlabeled_loader, test_loader, model, optimizer, scheduler, ema_model, patience):
    """
    Pseudo-Labeling baseline (--method pseudo): trains on labeled data with
    CE loss, and once the epoch passes args.T1, also trains on unlabeled data
    using the model's own argmax prediction as a hard label. The unlabeled
    loss weight (alpha) ramps up linearly from 0 to args.max_alpha between
    epochs T1 and T2, then stays fixed.
    """
    test_accs = []
    end = time.time()
    best_loss = float('inf')

    if args.world_size > 1:
        labeled_epoch = 0
        labeled_loader.sampler.set_epoch(labeled_epoch)
        unlabeled_epoch = 0
        unlabeled_loader.sampler.set_epoch(unlabeled_epoch)

    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)

    model.train()
    for epoch in range(args.epochs):
        batch_time = AverageMeter()
        losses = AverageMeter()

        if not args.no_progress:
            p_bar = tqdm(range(args.train_iteration), disable=args.local_rank not in [-1, 0])

        if (epoch > args.T1) and (epoch < args.T2):
            alpha = args.max_alpha * (epoch - args.T1) / (args.T2 - args.T1)
        elif epoch >= args.T2:
            alpha = args.max_alpha
        else:
            alpha = 0

        for batch_idx in range(args.train_iteration):
            try:
                inputs_x, targets_x = next(labeled_iter)
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    labeled_loader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(labeled_loader)
                inputs_x, targets_x = next(labeled_iter)
            try:
                inputs_u, _ = next(unlabeled_iter)
            except:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_loader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_loader)
                inputs_u, _ = next(unlabeled_iter)

            inputs_x, targets_x = inputs_x.to(args.device), targets_x.to(args.device)
            inputs_u = inputs_u.to(args.device)

            _, outputs_x = model(inputs_x)
            if alpha > 0:
                _, outputs_u = model(inputs_u)

                outputs_u = torch.softmax(outputs_u.detach(), dim=-1)
                _, pred_u = torch.max(outputs_u.detach(), 1)

                loss = F.cross_entropy(outputs_x, targets_x, reduction='mean') + alpha * F.cross_entropy(outputs_u,
                                                                                                         pred_u,
                                                                                                         reduction='mean')
            else:
                loss = F.cross_entropy(outputs_x, targets_x, reduction='mean')

            loss.backward()
            losses.update(loss.item())
            optimizer.step()
            scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()
            optimizer.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                p_bar.set_description(
                    "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. Ir: {lr:.4f}. Batch: {bt:.3f}s. Loss: {loss:.4f}.".format(
                        epoch=epoch + 1,
                        epochs=args.epochs,
                        batch=batch_idx + 1,
                        iter=args.train_iteration,
                        lr=scheduler.get_last_lr()[0],
                        bt=batch_time.avg,
                        loss=losses.avg))
                p_bar.update()

        if not args.no_progress:
            p_bar.close()

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        test_loss, test_acc, test_f1 = test(args, test_loader, test_model)
        test_accs.append(test_acc)

        # A NaN test loss (e.g. from a diverged run) must never look like an
        # improvement: Python/NumPy NaN comparisons are always False, so a
        # bare `test_loss >= best_loss` would silently fall into the `else`
        # branch below and save a corrupted checkpoint.
        if math.isnan(test_loss) or test_loss >= best_loss:
            patience += 1
            if patience >= args.patience_limit:
                break
        else:
            best_loss = deepcopy(test_loss)
            patience = 0
            torch.save(model.state_dict(), f'./result/{run_id(args)}.pth')
        print(f"patience : {patience + 1}/{args.patience_limit}")


def HCAE_train(args, train_loader, unlabeled_loader, test_loader, model, optimizer, scheduler, ema_model, patience):
    """
    Hybrid convolutional autoencoder baseline (--method hcae): the model
    (CAE_128/CAE_64) is trained with a classification CE loss on labeled data
    plus a reconstruction MSE loss on unlabeled data, so the encoder is
    shaped by both signals. Like Pseudo_train, the reconstruction loss
    weight (alpha) ramps up linearly between epochs T1 and T2.
    """
    test_accs = []
    end = time.time()
    best_loss = float('inf')

    if args.world_size > 1:
        labeled_epoch = 0
        train_loader.sampler.set_epoch(labeled_epoch)
        unlabeled_epoch = 0
        unlabeled_loader.sampler.set_epoch(unlabeled_epoch)

    labeled_iter = iter(train_loader)
    unlabeled_iter = iter(unlabeled_loader)

    model.train()
    for epoch in range(args.epochs):
        batch_time = AverageMeter()
        losses = AverageMeter()

        if not args.no_progress:
            p_bar = tqdm(range(args.train_iteration),
                         disable=args.local_rank not in [-1, 0])

        if (epoch > args.T1) and (epoch < args.T2):
            alpha = args.max_alpha * (epoch - args.T1) / (args.T2 - args.T1)
        elif epoch >= args.T2:
            alpha = args.max_alpha
        else:
            alpha = 0

        for batch_idx in range(args.train_iteration):
            try:
                inputs_x, targets_x = next(labeled_iter)
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    train_loader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(train_loader)
                inputs_x, targets_x = next(labeled_iter)

            try:
                inputs_u, _ = next(unlabeled_iter)
            except:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_loader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_loader)
                inputs_u, _ = next(unlabeled_iter)

            inputs_x, targets_x = inputs_x.to(args.device), targets_x.to(args.device)
            inputs_u = inputs_u.to(args.device)

            recons, _ = model(inputs_u)
            recon_loss = F.mse_loss(inputs_u, recons)

            _, outputs = model(inputs_x)

            cls_loss = F.cross_entropy(outputs, targets_x, reduction='mean')

            if alpha > 0:
                loss = cls_loss + alpha * recon_loss
            else:
                loss = cls_loss

            loss.backward()
            losses.update(loss.item())
            optimizer.step()
            scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()
            optimizer.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                p_bar.set_description(
                    "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. Ir: {lr:.4f}. Batch: {bt:.3f}s. Loss: {loss:.4f}.".format(
                        epoch=epoch + 1,
                        epochs=args.epochs,
                        batch=batch_idx + 1,
                        iter=args.train_iteration,
                        lr=scheduler.get_last_lr()[0],
                        bt=batch_time.avg,
                        loss=losses.avg))
                p_bar.update()

        if not args.no_progress:
            p_bar.close()

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        test_loss, test_acc, test_f1 = test(args, test_loader, test_model)
        test_accs.append(test_acc)

        # A NaN test loss (e.g. from a diverged run) must never look like an
        # improvement: Python/NumPy NaN comparisons are always False, so a
        # bare `test_loss >= best_loss` would silently fall into the `else`
        # branch below and save a corrupted checkpoint.
        if math.isnan(test_loss) or test_loss >= best_loss:
            patience += 1
            if patience >= args.patience_limit:
                break
        else:
            best_loss = deepcopy(test_loss)
            patience = 0
            torch.save(model.state_dict(), f'./result/{run_id(args)}.pth')
        print(f"patience : {patience + 1}/{args.patience_limit}")


def MixMatch_train(args, train_loader, unlabeled_loader, test_loader, model, criterion, optimizer, scheduler, ema_model,
                   patience):
    """
    MixMatch (--method mixmatch): averages predictions over two weak-aug
    views of each unlabeled sample, sharpens the average with temperature
    args.T to form soft pseudo-labels, then mixes (mixup) labeled and
    pseudo-labeled unlabeled samples together before computing the loss
    (criterion=MixLoss). interleave()/interleave_offset() keep BatchNorm
    statistics consistent when the mixed batches are split across multiple
    forward passes.
    """
    test_accs = []
    end = time.time()
    best_loss = float('inf')

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        train_loader.sampler.set_epoch(labeled_epoch)
        unlabeled_loader.sampler.set_epoch(unlabeled_epoch)

    labeled_iter = iter(train_loader)
    unlabeled_iter = iter(unlabeled_loader)

    model.train()
    for epoch in range(args.epochs):
        batch_time = AverageMeter()
        losses = AverageMeter()

        if not args.no_progress:
            p_bar = tqdm(range(args.train_iteration), disable=args.local_rank not in [-1, 0])

        for batch_idx in range(args.train_iteration):
            try:
                inputs_x, targets_x = next(labeled_iter)
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    train_loader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(train_loader)
                inputs_x, targets_x = next(labeled_iter)

            try:
                (inputs_u, inputs_u2), _ = next(unlabeled_iter)
            except:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_loader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_loader)
                (inputs_u, inputs_u2), _ = next(unlabeled_iter)

            batch_size = inputs_x.size(0)

            targets_x = torch.zeros(batch_size, args.num_classes).scatter_(1, targets_x.view(-1, 1).long(), 1)

            inputs_x, targets_x = inputs_x.to(args.device), targets_x.to(args.device)
            inputs_u, inputs_u2 = inputs_u.to(args.device), inputs_u2.to(args.device)

            with torch.no_grad():
                _, outputs = model(inputs_u)
                _, outputs_2 = model(inputs_u2)

                p = (torch.softmax(outputs, dim=1) + torch.softmax(outputs_2, dim=1)) / 2
                pt = p ** (1 / args.T)

                targets_u = pt / pt.sum(dim=1, keepdim=True)
                targets_u = targets_u.detach()

            all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2], dim=0)
            all_targets = torch.cat([targets_x, targets_u, targets_u], dim=0)

            l = np.random.beta(args.alpha, args.alpha)
            l = max(l, 1 - l)

            idx = torch.randperm(all_inputs.size(0))

            input_a, input_b = all_inputs, all_inputs[idx]
            target_a, target_b = all_targets, all_targets[idx]

            mixed_input = l * input_a + (1 - l) * input_b
            mixed_target = l * target_a + (1 - l) * target_b

            mixed_input = list(torch.split(mixed_input, batch_size))
            mixed_input = interleave(mixed_input, batch_size)

            logits = [model(mixed_input[0])[1]]
            for input in mixed_input[1:]:
                logits.append(model(input)[1])

            logits = interleave(logits, batch_size)
            logits_x = logits[0]
            logits_u = torch.cat(logits[1:], dim=0)

            Lx, Lu, w = criterion(logits_x, mixed_target[:batch_size], logits_u, mixed_target[batch_size:], epoch, args)
            loss = Lx + w * Lu

            losses.update(loss.item(), inputs_x.size(0))
            loss.backward()
            optimizer.step()
            scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()
            optimizer.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                p_bar.set_description(
                    "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. Ir: {lr:.4f}. Batch: {bt:.3f}s. Loss: {loss:.4f}.".format(
                        epoch=epoch + 1,
                        epochs=args.epochs,
                        batch=batch_idx + 1,
                        iter=args.train_iteration,
                        lr=scheduler.get_last_lr()[0],
                        bt=batch_time.avg,
                        loss=losses.avg))
                p_bar.update()

        if not args.no_progress:
            p_bar.close()

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        if args.local_rank in [-1, 0]:
            test_loss, test_acc, test_f1 = test(args, test_loader, test_model)

            test_accs.append(test_acc)
            # A NaN test loss (e.g. from a diverged run) must never look
            # like an improvement: Python/NumPy NaN comparisons are always
            # False, so a bare `test_loss >= best_loss` would silently fall
            # into the `else` branch below and save a corrupted checkpoint.
            if math.isnan(test_loss) or test_loss >= best_loss:
                patience += 1
                if patience >= args.patience_limit:
                    break
            else:
                best_loss = deepcopy(test_loss)
                patience = 0
                torch.save(model.state_dict(), f'./result/{run_id(args)}.pth')


def FixMatch_train(args, train_loader, unlabeled_loader, test_loader, model, optimizer, scheduler, ema_model, patience):
    """
    FixMatch (--method fixmatch): a weak-augmented view of each unlabeled
    sample produces a hard pseudo-label, kept only when its confidence
    exceeds args.threshold (masked). The strong-augmented view is then
    trained to predict that pseudo-label with cross-entropy, weighted by
    args.lambda_u.
    """
    test_accs = []
    best_loss = float('inf')
    end = time.time()

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        train_loader.sampler.set_epoch(labeled_epoch)
        unlabeled_loader.sampler.set_epoch(unlabeled_epoch)

    labeled_iter = iter(train_loader)
    unlabeled_iter = iter(unlabeled_loader)

    model.train()
    for epoch in range(args.epochs):
        batch_time = AverageMeter()
        losses = AverageMeter()
        mask_probs = AverageMeter()

        if not args.no_progress:
            p_bar = tqdm(range(args.train_iteration), disable=args.local_rank not in [-1, 0])

        for batch_idx in range(args.train_iteration):
            try:
                inputs_x, targets_x = next(labeled_iter)
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    train_loader.sampler.set_epoch(labeled_iter)
                labeled_iter = iter(train_loader)
                inputs_x, targets_x = next(labeled_iter)
            try:
                (inputs_w, inputs_s), _ = next(unlabeled_iter)
            except:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_loader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_loader)
                (inputs_w, inputs_s), _ = next(unlabeled_iter)

            batch_size = inputs_x.size(0)
            inputs = torch.cat((inputs_x, inputs_w, inputs_s)).to(args.device)
            targets_x = targets_x.to(args.device)

            _, outputs = model(inputs)
            outputs_x = outputs[:batch_size]
            outputs_w, outputs_s = outputs[batch_size:].chunk(2)
            del outputs

            Lx = F.cross_entropy(outputs_x, targets_x, reduction='mean')

            pseudo_label = torch.softmax(outputs_w.detach(), dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold).float()

            Lu = (F.cross_entropy(outputs_s, targets_u, reduction='mean') * mask).mean()

            loss = Lx + args.lambda_u * Lu

            losses.update(loss.item())
            loss.backward()
            optimizer.step()
            scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()
            optimizer.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()
            mask_probs.update(mask.mean().item())
            if not args.no_progress:
                p_bar.set_description(
                    "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. Ir: {lr:.4f}. Batch: {bt:.3f}s. Loss: {loss:.4f}. Mask: {mask:.2f}.".format(
                        epoch=epoch + 1,
                        epochs=args.epochs,
                        batch=batch_idx + 1,
                        iter=args.train_iteration,
                        lr=scheduler.get_last_lr()[0],
                        bt=batch_time.avg,
                        loss=losses.avg,
                        mask=mask_probs.avg))
                p_bar.update()

        if not args.no_progress:
            p_bar.close()

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        if args.local_rank in [-1, 0]:
            test_loss, test_acc, test_f1 = test(args, test_loader, test_model)

            test_accs.append(test_acc)

            # A NaN test loss (e.g. from a diverged run) must never look
            # like an improvement: Python/NumPy NaN comparisons are always
            # False, so a bare `test_loss >= best_loss` would silently fall
            # into the `else` branch below and save a corrupted checkpoint.
            if math.isnan(test_loss) or test_loss >= best_loss:
                patience += 1
                if patience >= args.patience_limit:
                    break
            else:
                best_loss = deepcopy(test_loss)
                patience = 0
                torch.save(model.state_dict(), f'./result/{run_id(args)}.pth')
            print(f"patience : {patience + 1}/{args.patience_limit}")



def SimMatch_train(args, train_loader, unlabeled_loader, test_loader, model, optimizer, scheduler, ema_model, patience):
    """
    SimMatch (--method simmatch): combines three consistency signals between
    a weak- and strong-augmented view of each unlabeled sample —
        - L_sem: classifier (semantic) pseudo-label agreement, FixMatch-style
        - L_graph: each view's feature-similarity graph is pulled toward the
          label-agreement graph built from the weak view's pseudo-labels
        - L_inst: instance-level agreement — class labels are propagated
          from a FeatureMemory bank of past labeled embeddings to each view
          by similarity (instance_pseudo_label), and the two views' resulting
          soft labels must match
    weighted by args.lambda_s / args.lambda_g / args.lambda_i respectively
    (all default to 1.0 via getattr since they aren't CLI args).
    """
    test_accs = []
    best_loss = float("inf")
    end = time.time()
    device = args.device

    mem_bank = FeatureMemory(capacity=getattr(args, "feature_mem", 4096), device=device)

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        train_loader.sampler.set_epoch(labeled_epoch)
        unlabeled_loader.sampler.set_epoch(unlabeled_epoch)

    labeled_iter = iter(train_loader)
    unlabeled_iter = iter(unlabeled_loader)

    model.train()
    for epoch in range(args.epochs):
        batch_time = AverageMeter()
        losses = AverageMeter()
        mask_probs = AverageMeter()

        if not args.no_progress and args.local_rank in [-1, 0]:
            p_bar = tqdm(range(args.train_iteration), disable=False, leave=False)

        for batch_idx in range(args.train_iteration):
            try:
                inputs_x, targets_x = next(labeled_iter)
            except StopIteration:
                if args.world_size > 1:
                    labeled_epoch += 1
                    train_loader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(train_loader)
                inputs_x, targets_x = next(labeled_iter)

            try:
                (inputs_w, inputs_s), _ = next(unlabeled_iter)
            except StopIteration:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_loader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_loader)
                (inputs_w, inputs_s), _ = next(unlabeled_iter)

            batch_size = inputs_x.size(0)

            all_inputs = torch.cat((inputs_x, inputs_w, inputs_s)).to(args.device)
            targets_x = targets_x.to(args.device)

            logits, probs = model(all_inputs)

            probs_x = probs[:batch_size]
            probs_w, probs_s = probs[batch_size:].chunk(2)
            logits_w, logits_s = logits[batch_size:].chunk(2)

            Lx = F.cross_entropy(probs_x, targets_x, reduction='mean')

            pseudo_prob = torch.softmax(probs_w.detach(), dim=-1)
            max_probs, targets_u = torch.max(pseudo_prob, dim=-1)
            mask = max_probs.ge(args.threshold).float()

            L_sem = (F.cross_entropy(probs_s, targets_u, reduction='none') * mask).mean()

            # Label-agreement graph from the shared pseudo-labels, used as a
            # fixed target for each view's own feature-similarity graph so
            # that same-pseudo-label samples are pulled toward similar
            # features (previously Gw was compared to Gs, which are built
            # from the same targets_u and are therefore always identical —
            # that term always evaluated to zero and taught the model nothing).
            G = build_semantic_graph(targets_u)

            Hw = build_similarity(logits_w)
            Hs = build_similarity(logits_s)

            graph_k = getattr(args, "graph_k", None)
            if graph_k is not None and graph_k > 0:
                def _topk(mat):
                    topk, _ = torch.topk(mat, graph_k, dim=1)
                    thresh = topk[:, -1].unsqueeze(1)
                    return mat * (mat >= thresh).float()

                Hw = _topk(Hw)
                Hs = _topk(Hs)

            L_graph = mse_detach(G, Hw) + mse_detach(G, Hs)
            conf_mask = mask.unsqueeze(1) * mask.unsqueeze(0)
            L_graph = (L_graph * conf_mask).sum() / (conf_mask.sum() + 1e-6)

            # Instance-level consistency (SimMatch): propagate class labels
            # from the FeatureMemory bank of past labeled embeddings to the
            # weak/strong embeddings by similarity, then require the strong
            # view's instance pseudo-label to match the weak view's. This is
            # the piece that was missing before — mem_bank was populated via
            # enqueue() every iteration but never read back with get().
            mem_feats, mem_labels = mem_bank.get()
            if mem_feats is not None:
                inst_tau = getattr(args, "inst_tau", 0.1)
                inst_w = instance_pseudo_label(logits_w, mem_feats, mem_labels, args.num_classes, tau=inst_tau)
                inst_s = instance_pseudo_label(logits_s, mem_feats, mem_labels, args.num_classes, tau=inst_tau)
                L_inst = (F.mse_loss(inst_w.detach(), inst_s, reduction='none').sum(dim=1) * mask).mean()
            else:
                L_inst = torch.zeros((), device=args.device)

            loss = Lx + getattr(args, "lambda_s", 1.0) * L_sem + \
                   getattr(args, "lambda_g", 1.0) * L_graph + \
                   getattr(args, "lambda_i", 1.0) * L_inst

            losses.update(loss.item())
            mask_probs.update(mask.mean().item())

            loss.backward()
            optimizer.step();
            scheduler.step()

            if args.use_ema:
                ema_model.update(model)

            model.zero_grad();
            optimizer.zero_grad()

            batch_time.update(time.time() - end);
            end = time.time()

            if not args.no_progress and args.local_rank in [-1, 0]:
                p_bar.set_description(
                    "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. Ir: {lr:.4f}. Batch: {bt:.3f}s. Loss: {loss:.4f}. Mask: {mask:.2f}".format(
                        epoch=epoch + 1,
                        epochs=args.epochs,
                        batch=batch_idx + 1,
                        iter=args.train_iteration,
                        lr=scheduler.get_last_lr()[0],
                        bt=batch_time.avg,
                        loss=losses.avg,
                        mask=mask_probs.avg))
                p_bar.update()

            mem_bank.enqueue(feat=logits[:batch_size].detach(), label=targets_x.detach())

        if not args.no_progress and args.local_rank in [-1, 0]:
            p_bar.close()

        test_model = ema_model.ema if args.use_ema else model
        if args.local_rank in [-1, 0]:
            test_loss, test_acc, test_f1 = test(args, test_loader, test_model)
            test_accs.append(test_acc)

            # A NaN test loss (e.g. from a diverged run) must never look
            # like an improvement: Python/NumPy NaN comparisons are always
            # False, so a bare `test_loss >= best_loss` would silently fall
            # into the `else` branch below and save a corrupted checkpoint.
            if math.isnan(test_loss) or test_loss >= best_loss:
                patience += 1
                if patience >= args.patience_limit:
                    break
            else:
                best_loss = deepcopy(test_loss)
                patience = 0
                torch.save(model.state_dict(), f"./result/{run_id(args)}.pth")
            print(f"patience : {patience + 1}/{args.patience_limit}")


def Proposed_train(args, train_loader, unlabeled_loader, test_loader, model, optimizer, scheduler, ema_model, patience):
    """
    This paper's method (--method proposed). Labeled data gets n_aug
    time-masked views (Transform_Proposed_Multi) plus within-class mixup;
    unlabeled data gets stronger SpecAugment views (Transform_Proposed_ulb).
    Unlabeled predictions above args.threshold are hardened to one-hot,
    otherwise kept soft. The loss combines labeled CE, an entropy penalty on
    unlabeled predictions (calculate_entropy, encourages confident
    predictions), and a soft triplet loss (batch_triplet_loss) that pulls
    each labeled anchor's feature toward unlabeled samples with high
    predicted probability for the anchor's class and away from the rest.

    Ablation flags (all default off, i.e. this docstring's behavior):
    args.no_mask_aug swaps the masking transforms for Transform_Proposed_NoAug
    (see data_loader.get_transform), args.no_mixup skips the mixup call
    below, and args.no_triplet / args.no_entropy each drop their loss term
    (CE always stays).
    """
    test_accs = []
    end = time.time()
    best_loss = float('inf')

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        train_loader.sampler.set_epoch(labeled_epoch)
        unlabeled_loader.sampler.set_epoch(unlabeled_epoch)

    labeled_iter = iter(train_loader)
    unlabeled_iter = iter(unlabeled_loader)

    model.train()
    for epoch in range(args.epochs):
        batch_time = AverageMeter()
        losses = AverageMeter()

        if not args.no_progress:
            p_bar = tqdm(range(args.train_iteration), disable=args.local_rank not in [-1, 0])

        if (epoch > args.T1) and (epoch < args.T2):
            alpha = args.max_alpha * (epoch - args.T1) / (args.T2 - args.T1)
        elif epoch >= args.T2:
            alpha = args.max_alpha
        else:
            alpha = 0

        for batch_idx in range(args.train_iteration):
            try:
                aug_list, target_x = next(labeled_iter)
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    train_loader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(train_loader)

                aug_list, target_x = next(labeled_iter)

            target_x = target_x.long()

            try:
                unaug_list, _ = next(unlabeled_iter)
            except:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_loader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_loader)
                unaug_list, _ = next(unlabeled_iter)

                
            inputs_x = torch.cat(aug_list, dim=0).to(args.device)
            targets_x = torch.cat([target_x for _ in range(len(aug_list))], dim=0).to(args.device)

            inputs_u = torch.cat(unaug_list, dim=0).to(args.device)

            if not getattr(args, 'no_mixup', False):
                if getattr(args, 'legacy_mixup', False):
                    inputs_x, targets_x = mixup_data_within_cls_legacy(inputs_x, targets_x)
                else:
                    inputs_x, targets_x = mixup_data_within_cls(inputs_x, targets_x)

            logits_x, probs_x = model(inputs_x)
            logits_u, probs_u = model(inputs_u)
            probs_u = F.softmax(probs_u, dim=1)

            with torch.no_grad():
                p_max, p_label = probs_u.max(dim=1)
                mask = p_max.ge(args.threshold)

                p_hard = F.one_hot(p_label, num_classes=args.num_classes).float()

                probs_u = torch.where(mask.unsqueeze(1), p_hard, probs_u)

            labeled_loss = F.cross_entropy(probs_x, targets_x, reduction='mean')
            loss = labeled_loss

            if not getattr(args, 'no_entropy', False):
                entropy_u = calculate_entropy(probs_u).mean()
                loss = loss + entropy_u

            if not getattr(args, 'no_triplet', False):
                # skip the (relatively expensive) triplet computation entirely when disabled
                if getattr(args, 'legacy_margin', False):
                    triplet_loss = batch_triplet_loss_legacy(args, logits_x, targets_x, logits_u, probs_u)
                else:
                    triplet_loss = batch_triplet_loss(args, logits_x, targets_x, logits_u, probs_u)
                loss = loss + triplet_loss

            losses.update(loss.item())
            loss.backward()
            optimizer.step()
            scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()
            optimizer.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                p_bar.set_description(
                    "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. Ir: {lr:.4f}. Batch: {bt:.3f}s. Loss: {loss:.4f}.".format(
                        epoch=epoch + 1,
                        epochs=args.epochs,
                        batch=batch_idx + 1,
                        iter=args.train_iteration,
                        lr=scheduler.get_last_lr()[0],
                        bt=batch_time.avg,
                        loss=losses.avg))
                p_bar.update()

        if not args.no_progress:
            p_bar.close()

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        if args.local_rank in [-1, 0]:
            test_loss, test_acc, test_f1 = test(args, test_loader, test_model)

            test_accs.append(test_acc)

            # A NaN test loss (e.g. from a diverged run) must never look
            # like an improvement: Python/NumPy NaN comparisons are always
            # False, so a bare `test_loss >= best_loss` would silently fall
            # into the `else` branch below and save a corrupted checkpoint.
            if math.isnan(test_loss) or test_loss >= best_loss:
                patience += 1
                if patience >= args.patience_limit:
                    break
            else:
                best_loss = deepcopy(test_loss)
                patience = 0
                torch.save(model.state_dict(), f'./result/{run_id(args)}.pth')
            print(f"patience : {patience + 1}/{args.patience_limit}")
