import time
from copy import deepcopy
import torch.nn as nn
import torch.nn.functional as F

from utils import *


def _supervised_train(args, trainloader, testloader, model, optimizer, scheduler, ema_model, patience):
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

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs = inputs.to(args.device)
            targets = targets.long()
            targets = targets.to(args.device)

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

            if test_loss >= best_loss:
                patience += 1
                if patience >= args.patience_limit:
                    break
            else:
                best_loss = deepcopy(test_loss)
                patience = 0
                torch.save(model.state_dict(), f'./result/{args.model}_{args.method}_{args.num_labeled}_{args.dataset}.pth')
            print(f"patience : {patience + 1}/{args.patience_limit}")

def supervised_train(args, trainloader, testloader, model, optimizer, scheduler, ema_model, patience):
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

            if test_loss >= best_loss:
                patience += 1
                if patience >= args.patience_limit:
                    break
            else:
                best_loss = deepcopy(test_loss)
                patience = 0
                torch.save(model.state_dict(), f'./result/{args.model}_{args.method}_{args.num_labeled}_{args.dataset}.pth')
            print(f"patience : {patience + 1}/{args.patience_limit}")

def Pseudo_train(args, labeled_loader, unlabeled_loader, test_loader, model, optimizer, scheduler, ema_model, patience):
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

        test_loss, test_acc = test(args, test_loader, test_model)
        test_accs.append(test_acc)

        if test_loss >= best_loss:
            patience += 1
            if patience >= args.patience_limit:
                break
        else:
            best_loss = deepcopy(test_loss)
            patience = 0
            torch.save(model.state_dict(), f'./result/{args.model}_{args.method}_{args.num_labeled}.pth')
        print(f"patience : {patience + 1}/{args.patience_limit}")


def HCAE_train(args, train_loader, unlabeled_loader, test_loader, model, optimizer, scheduler, ema_model, patience):
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

        test_loss, test_acc = test(args, test_loader, test_model)
        test_accs.append(test_acc)

        if test_loss >= best_loss:
            patience += 1
            if patience >= args.patience_limit:
                break
        else:
            best_loss = deepcopy(test_loss)
            patience = 0
            torch.save(model.state_dict(), f'./result/{args.model}_{args.method}_{args.num_labeled}.pth')
        print(f"patience : {patience + 1}/{args.patience_limit}")


def MixMatch_train(args, train_loader, unlabeled_loader, test_loader, model, criterion, optimizer, scheduler, ema_model,
                   patience):
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

            targets_x = torch.zeros(batch_size, 7).scatter_(1, targets_x.view(-1, 1).long(), 1)

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
            test_loss, test_acc = test(args, test_loader, test_model)

            test_accs.append(test_acc)
            if test_loss >= best_loss:
                patience += 1
                if patience >= args.patience_limit:
                    break
            else:
                best_loss = deepcopy(test_loss)
                patience = 0
                torch.save(model.state_dict(), f'./result/{args.model}_{args.method}_{args.num_labeled}.pth')


def FixMatch_train(args, train_loader, unlabeled_loader, test_loader, model, optimizer, scheduler, ema_model, patience):
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
            # inputs = fix_interleave(torch.cat((inputs_x, inputs_w, inputs_s)), 2 * args.mu + 1).to(args.device)
            inputs = torch.cat((inputs_x, inputs_w, inputs_s)).to(args.device)
            targets_x = targets_x.to(args.device)

            _, outputs = model(inputs)
            # outputs = de_interleave(outputs, 2 * args.mu + 1)
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
            test_loss, test_acc = test(args, test_loader, test_model)

            test_accs.append(test_acc)

            if test_loss >= best_loss:
                patience += 1
                if patience >= args.patience_limit:
                    break
            else:
                best_loss = deepcopy(test_loss)
                patience = 0
                torch.save(model.state_dict(), f'./result/{args.model}_{args.method}_{args.num_labeled}.pth')
            print(f"patience : {patience + 1}/{args.patience_limit}")



def SimMatch_train(args, train_loader, unlabeled_loader, test_loader, model, optimizer, scheduler, ema_model, patience):
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

            # Interleave for SyncBN
            all_inputs = torch.cat((inputs_x, inputs_w, inputs_s)).to(args.device)
            # all_inputs = fix_interleave(all_inputs, 2 * args.mu + 1).to(args.device)
            targets_x = targets_x.to(args.device)

            # Model expected to output
            logits, probs = model(all_inputs)
            # logits = de_interleave(logits, 2 * args.mu + 1)
            # probs = de_interleave(probs, 2 * args.mu + 1)

            probs_x = probs[:batch_size]
            probs_w, probs_s = probs[batch_size:].chunk(2)
            logits_w, logits_s = logits[batch_size:].chunk(2)

            Lx = F.cross_entropy(probs_x, targets_x, reduction='mean')

            pseudo_prob = torch.softmax(probs_w.detach(), dim=-1)
            max_probs, targets_u = torch.max(pseudo_prob, dim=-1)
            mask = max_probs.ge(args.threshold).float()

            L_sem = (F.cross_entropy(probs_s, targets_u, reduction='none') * mask).mean()

            Gw = build_semantic_graph(targets_u)
            Gs = build_semantic_graph(targets_u)

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

            L_graph = mse_detach(Gw, Gs) + mse_detach(Hw, Hs)
            conf_mask = mask.unsqueeze(1) * mask.unsqueeze(0)
            L_graph = (L_graph * conf_mask).sum() / (conf_mask.sum() + 1e-6)

            loss = Lx + getattr(args, "lambda_s", 1.0) * L_sem + \
                   getattr(args, "lambda_g", 1.0) * L_graph

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
            test_loss, test_acc = test(args, test_loader, test_model)
            test_accs.append(test_acc)

            if test_loss >= best_loss:
                patience += 1
                if patience >= args.patience_limit:
                    break
            else:
                best_loss = deepcopy(test_loss)
                patience = 0
                torch.save(model.state_dict(), f"./result/{args.model}_{args.method}_{args.num_labeled}.pth")
            print(f"patience : {patience + 1}/{args.patience_limit}")


def Proposed_train(args, train_loader, unlabeled_loader, test_loader, model, optimizer, scheduler, ema_model, patience):
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
            entropy_u = calculate_entropy(probs_u).mean()

            triplet_loss = batch_triplet_loss(args, logits_x, targets_x, logits_u, probs_u)

            loss = labeled_loss + triplet_loss + entropy_u

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

            if test_loss >= best_loss:
                patience += 1
                if patience >= args.patience_limit:
                    break
            else:
                best_loss = deepcopy(test_loss)
                patience = 0
                torch.save(model.state_dict(), f'./result/{args.model}_{args.method}_{args.num_labeled}_{args.dataset}.pth')
            print(f"patience : {patience + 1}/{args.patience_limit}")
