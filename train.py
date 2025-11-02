
# train.py - training loop for segmentation model
import os, json, argparse
import torch
from torch.utils.data import DataLoader
from dataset import NCCT2p5DDataset
from model import UNet2p5D
import torch.nn as nn
import numpy as np

def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = 1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth))
    return loss.mean()

def train_epoch(model, loader, opt, device):
    model.train()
    total_loss = 0.0
    for x,y in loader:
        x = x.to(device)
        y = y.to(device)
        opt.zero_grad()
        out = model(x)
        bce = nn.BCELoss()(out, y)
        dloss = dice_loss(out, y)
        loss = bce + dloss
        loss.backward()
        opt.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def val_epoch(model, loader, device):
    model.eval()
    tot_dice = 0.0
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            pred = (out>0.5).float()
            intersection = (pred * y).sum()
            dice = (2. * intersection) / (pred.sum() + y.sum() + 1e-6)
            tot_dice += dice.item()
    return tot_dice / len(loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.json')
    args = parser.parse_args()
    cfg = json.load(open(args.config))
    os.makedirs(cfg['paths']['checkpoints'], exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    import pandas as pd
    train_df = pd.read_csv(cfg['paths']['train_list'])
    val_df = pd.read_csv(cfg['paths']['val_list'])
    train_paths = train_df['image'].tolist()
    train_masks = train_df['mask'].tolist()
    val_paths = val_df['image'].tolist()
    val_masks = val_df['mask'].tolist()
    train_ds = NCCT2p5DDataset(train_paths, train_masks, num_slices=cfg['train']['num_slices'],
                               patch_size=tuple(cfg['train']['patch_size']), transforms=None)
    val_ds = NCCT2p5DDataset(val_paths, val_masks, num_slices=cfg['train']['num_slices'],
                             patch_size=tuple(cfg['train']['patch_size']), transforms=None)
    train_loader = DataLoader(train_ds, batch_size=cfg['train']['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1)
    model = UNet2p5D(in_channels=cfg['train']['num_slices']).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['train']['lr'])
    best = 0.0
    for epoch in range(cfg['train']['epochs']):
        tr_loss = train_epoch(model, train_loader, opt, device)
        val_dice = val_epoch(model, val_loader, device)
        print(f"Epoch {epoch} TrainLoss {tr_loss:.4f} ValDice {val_dice:.4f}")
        if val_dice > best:
            best = val_dice
            torch.save(model.state_dict(), os.path.join(cfg['paths']['checkpoints'], 'best.pth'))
