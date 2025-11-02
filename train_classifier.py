
# train_classifier.py - training loop for encoder classifier
import os, json, argparse
import torch
from torch.utils.data import DataLoader, Dataset
from classifier import EncoderClassifier
import torch.nn as nn
import pandas as pd
import nibabel as nib
import numpy as np

class SimpleClassDataset(Dataset):
    def __init__(self, df_list, num_slices=5):
        self.df_list = df_list
        self.num_slices = num_slices

    def __len__(self):
        return len(self.df_list)

    def __getitem__(self, idx):
        row = self.df_list.iloc[idx]
        img = nib.load(row['image']).get_fdata()
        z,y,x = img.shape
        mid = z//2
        half = self.num_slices//2
        zs = np.clip(np.arange(mid-half, mid+half+1), 0, z-1)
        slices = img[zs,:,:]
        inp = np.stack(slices, axis=0).astype('float32')
        label = int(row['label']) if 'label' in row else 0
        return torch.from_numpy(inp), torch.tensor(label, dtype=torch.long)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.json')
    args = parser.parse_args()
    cfg = json.load(open(args.config))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # CSV expected to have columns: image, label (0 normal, 1 abnormal)
    train_df = pd.read_csv('data/splits/classifier_train.csv')
    val_df = pd.read_csv('data/splits/classifier_val.csv')
    train_ds = SimpleClassDataset(train_df, num_slices=cfg['train']['num_slices'])
    val_ds = SimpleClassDataset(val_df, num_slices=cfg['train']['num_slices'])
    train_loader = DataLoader(train_ds, batch_size=cfg['train']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    model = EncoderClassifier(in_channels=cfg['train']['num_slices']).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['train']['lr'])
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    for epoch in range(cfg['train']['epochs']):
        model.train()
        total=0; correct=0
        for x,y in train_loader:
            x=x.to(device); y=y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward(); opt.step()
        # val
        model.eval()
        total=0; correct=0
        with torch.no_grad():
            for x,y in val_loader:
                x=x.to(device); y=y.to(device)
                out = model(x)
                pred = out.argmax(dim=1)
                total+=1; correct += (pred==y).sum().item()
        acc = correct/total if total>0 else 0
        print(f"Epoch {epoch} ValAcc {acc:.3f}")
        if acc>best_acc:
            best_acc=acc
            os.makedirs(cfg['paths']['classifier_checkpoints'], exist_ok=True)
            torch.save(model.state_dict(), os.path.join(cfg['paths']['classifier_checkpoints'], 'best_classifier.pth'))
