import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from .losses import compute_loss_with_criterion

# from .utils import TimeElapsed

# __all__ = []


def train_multistep(model, loader, preprocess, optimizer, scheduler, amp_scaler, config, steps):
    model.train()

    i = 0
    cumu_loss = 0
    correct, total = (0, 0)

    while True:
        for sample_batched in loader:
            optimizer.zero_grad()

            # preprocessing (this includes to-device operation)
            preprocess(sample_batched)

            # pull the data
            x = sample_batched["signal"]
            age = sample_batched["age"]
            y = sample_batched["class_label"]

            # mixed precision training if needed
            with autocast('cuda', enabled=config.get("mixed_precision", False)):
                # forward pass
                if hasattr(model, 'use_age') and model.use_age == "no":
                    output = model(x)  # Don't pass age if not used
                else:
                    output = model(x, age)  # Pass age for backward compatibility

                # Get class weights if available
                class_weights = config.get("class_weights", None)
                if class_weights is not None:
                    class_weights = class_weights.to(output.device)

                # Compute loss using the new loss function system
                loss, s = compute_loss_with_criterion(
                    output, y, config["criterion"], 
                    class_weights=class_weights,
                    focal_alpha=config.get("focal_alpha", 1.0),
                    focal_gamma=config.get("focal_gamma", 2.0)
                )

            # backward and update
            if config.get("mixed_precision", False):
                amp_scaler.scale(loss).backward()
                if "clip_grad_norm" in config:
                    amp_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad_norm"])
                amp_scaler.step(optimizer)
                amp_scaler.update()
                scheduler.step()
            else:
                loss.backward()
                if "clip_grad_norm" in config:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad_norm"])
                optimizer.step()
                scheduler.step()

            # train accuracy
            pred = s.argmax(dim=-1)
            correct += pred.squeeze().eq(y).sum().item()
            total += pred.shape[0]
            cumu_loss += loss.item()

            i += 1
            if steps <= i:
                break
        if steps <= i:
            break

    train_acc = 100.0 * correct / total
    avg_loss = cumu_loss / steps

    return avg_loss, train_acc


def train_mixup_multistep(model, loader, preprocess, optimizer, scheduler, amp_scaler, config, steps):
    model.train()

    i = 0
    cumu_loss = 0
    correct, total = (0, 0)

    while True:
        for sample_batched in loader:
            optimizer.zero_grad()

            # preprocessing (this includes to-device operation)
            preprocess(sample_batched)

            # load and mixup the mini-batched data
            x1 = sample_batched["signal"]
            age1 = sample_batched["age"]
            y1 = sample_batched["class_label"]

            index = torch.randperm(x1.shape[0]).cuda()
            x2 = x1[index]
            age2 = age1[index]
            y2 = y1[index]

            mixup_alpha = config["mixup"]
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            x = lam * x1 + (1.0 - lam) * x2
            age = lam * age1 + (1.0 - lam) * age2

            # mixed precision training if needed
            with autocast('cuda', enabled=config.get("mixed_precision", False)):
                # forward pass
                if hasattr(model, 'use_age') and model.use_age == "no":
                    output = model(x)  # Don't pass age if not used
                else:
                    output = model(x, age)  # Pass age for backward compatibility

                # Get class weights if available
                class_weights = config.get("class_weights", None)
                if class_weights is not None:
                    class_weights = class_weights.to(output.device)

                # Compute mixup loss using the new loss function system
                loss1, s1 = compute_loss_with_criterion(
                    output, y1, config["criterion"], 
                    class_weights=class_weights,
                    focal_alpha=config.get("focal_alpha", 1.0),
                    focal_gamma=config.get("focal_gamma", 2.0)
                )
                loss2, s2 = compute_loss_with_criterion(
                    output, y2, config["criterion"], 
                    class_weights=class_weights,
                    focal_alpha=config.get("focal_alpha", 1.0),
                    focal_gamma=config.get("focal_gamma", 2.0)
                )
                
                loss = lam * loss1 + (1 - lam) * loss2
                s = s1  # Use s1 for prediction accuracy (could also interpolate)

            # backward and update
            if config.get("mixed_precision", False):
                amp_scaler.scale(loss).backward()
                if "clip_grad_norm" in config:
                    amp_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad_norm"])
                amp_scaler.step(optimizer)
                amp_scaler.update()
                scheduler.step()
            else:
                loss.backward()
                if "clip_grad_norm" in config:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad_norm"])
                optimizer.step()
                scheduler.step()

            # train accuracy
            pred = s.argmax(dim=-1)
            correct1 = pred.squeeze().eq(y1).sum().item()
            correct2 = pred.squeeze().eq(y2).sum().item()
            correct += lam * correct1 + (1.0 - lam) * correct2
            total += pred.shape[0]
            cumu_loss += loss.item()

            i += 1
            if steps <= i:
                break
        if steps <= i:
            break

    train_acc = 100.0 * correct / total
    avg_loss = cumu_loss / steps

    return avg_loss, train_acc
