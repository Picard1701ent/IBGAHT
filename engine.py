import logging
from types import SimpleNamespace

import config
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from models.IBAHGT import IBAHGT
from utils import load_datas

cfg = SimpleNamespace(**vars(config))
device = torch.device(cfg.device)


def evaluate_result(out, labels):
    prob = out
    preds = torch.argmax(prob, dim=1)
    probs_pos_class = prob[:, 1].detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    auc = roc_auc_score(labels_np, probs_pos_class)
    accuracy = accuracy_score(labels_np, preds.cpu().numpy())

    f1 = f1_score(labels_np, preds.cpu().numpy())
    tn, fp, fn, tp = confusion_matrix(labels_np, preds.cpu().numpy()).ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    correct_predictions = (preds.cpu().numpy() == labels_np).astype(int)

    return accuracy, f1, sensitivity, specificity, auc, correct_predictions


def train_one_epoch(model, optimizer, dataloader, loss_func):
    total_loss = 0.0
    total = 0
    total_out = []
    total_labels = []
    model.train()
    for datas, Hs, labels in dataloader:
        datas = datas.to(device)
        labels = labels.to(device)
        Hs = Hs.to(device)
        optimizer.zero_grad()
        out, loss0 = model(datas, Hs, labels)
        loss = loss_func(out, labels)
        loss = loss + cfg.alpha * loss0
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item()
        total += labels.shape[0]
        total_out.append(out)
        total_labels.append(labels)
    total_out = torch.cat(total_out, dim=0)
    total_labels = torch.cat(total_labels, dim=0)
    accuracy, f1, sensitivity, specificity, auc, correct_predictions = evaluate_result(
        total_out, total_labels
    )

    return accuracy, f1, sensitivity, specificity, auc, total_loss / total


def evaluate_model(model, dataloader, loss_fn):
    model.eval()
    total_loss = 0.0
    total = 0
    total_prob = []
    total_label = []
    total_attn_weights = []
    with torch.no_grad():
        for datas, Hs, labels in dataloader:
            datas = datas.to(device)
            labels = labels.to(device)
            Hs = Hs.to(device)
            out, loss0 = model(datas, Hs, labels)
            attn_weights = model.attn_weights
            loss = loss_fn(out, labels)
            loss = loss + loss0
            total_loss = loss.item() + total_loss
            total = total + datas.shape[0]
            total_prob.append(out)
            total_label.append(labels)
            total_attn_weights.append(attn_weights)
    total_prob = torch.cat(total_prob, dim=0)
    total_label = torch.cat(total_label, dim=0)
    accuracy, f1, sensitivity, specificity, auc, correct_predictions = evaluate_result(
        total_prob, total_label
    )
    return (
        accuracy,
        f1,
        sensitivity,
        specificity,
        auc,
        total_loss / total,
        total_attn_weights,
        total_label,
        correct_predictions,
    )


def train():
    datas, Hs, labels = load_datas()
    # labels, datas = get_dataset()
    datas = torch.from_numpy(datas).to(torch.float32)
    labels = torch.from_numpy(labels).to(torch.long)
    Hs = torch.from_numpy(Hs).to(torch.float32)
    print(labels.shape)
    dataset = TensorDataset(datas, Hs, labels)

    train_idx, val_idx = train_test_split(
        np.arange(len(datas)),
        test_size=0.2,
        stratify=labels.numpy(),
        random_state=cfg.seed,
    )
    logging.info(f"Train index: {train_idx}, Val index: {val_idx}")

    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=cfg.batch_size, shuffle=True)
    class_counts = torch.bincount(labels[train_idx])
    total_samples = len(labels[train_idx])

    weights = total_samples / class_counts.float()
    weights = weights.to(device)
    model = IBAHGT(num_layer=cfg.model_num_layer, weight=weights)
    loss_func = torch.nn.CrossEntropyLoss(weight=weights, reduction="mean")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epoch, eta_min=0)
    model.to(device)
    model.train()

    for i in range(cfg.epoch):
        accuracy, f1, sensitivity, specificity, auc, ave_loss = train_one_epoch(
            model, optimizer, train_loader, loss_func
        )
        scheduler.step()

        train_acc_percent = accuracy * 100

        logging.info(
            f"(Train) | Epoch={i:03d}, loss={ave_loss:.4f}, "
            f"train_acc={(accuracy * 100):.2f}, train_specificity={(specificity * 100):.2f}, "
            f"train_sensitivity={(sensitivity * 100):.2f},train_auc={(auc * 100):.2f}"
        )

        (
            accuracy,
            f1,
            sensitivity,
            specificity,
            auc,
            total_loss,
            eval_attn_weight,
            eval_labels,
            eval_correct,
        ) = evaluate_model(model, val_loader, loss_func)

        text = (
            f"(Evaluate) | Epoch={i:03d}, loss={total_loss:.4f}, "
            f"test_acc={(accuracy * 100):.2f}, test_specificity={(specificity * 100):.2f}, "
            f"test_sensitivity={(sensitivity * 100):.2f},test_auc={(auc * 100):.2f},f1={(f1 * 100):.2f}"
        )
        logging.info(text)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
