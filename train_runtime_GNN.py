import os, warnings, argparse, random, time
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader

try:
    from torch.amp import autocast, GradScaler   # PyTorch ≥ 2.3
except Exception:
    from torch.cuda.amp import autocast, GradScaler

from modules import (
    Average, stat,
    load_data_partial, load_data_PV, load_data_PV_multi_attack, load_data_PV_Attack_type,
    make_graph_data
)
from models_GNN import GNN, GraphTransformer, Graphormer, GPSModel

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*", category=FutureWarning)

# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# -----------------------------
# Argparse
# -----------------------------
parser = argparse.ArgumentParser(description='GPU-friendly PyG experiment (80/20 split + timing)')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--dataset', type=str, default='8500 bus')  # "34 bus", "123 bus", "8500 bus"
parser.add_argument('--PV_attack', type=str, default='partial_Oable',
                    choices=['SinglePV','MultiplePV','Atype_detection','partial_Oable'])
parser.add_argument('--model', type=str, default='GIN',
                    choices=['GCN','SAGE','GAT','GIN','UniMP','graphormer','GPS'])
parser.add_argument('--N_Scenarios', type=int, default=500)
parser.add_argument('--hidden_channels', type=int, default=64)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--amp', action='store_true')
parser.add_argument('--compile', action='store_true')
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()
print(args)

set_seed(args.seed)

# -----------------------------
# Device
# -----------------------------
if torch.cuda.is_available():
    ndev = torch.cuda.device_count()
    if args.device >= ndev:
        print(f"[WARN] Requested device {args.device} not available (found {ndev}). Using 0.")
        args.device = 0
    torch.cuda.set_device(args.device)
    device = torch.device(f'cuda:{args.device}')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try: torch.set_float32_matmul_precision('high')
    except Exception: pass
else:
    device = torch.device('cpu')
print(f"Using device: {device}")

# -----------------------------
# Data loading
# -----------------------------
def load_dataset(kind, dataset, n):
    if kind == 'SinglePV':      return load_data_PV(dataset, n)
    if kind == 'MultiplePV':    return load_data_PV_multi_attack(dataset, n)
    if kind == 'Atype_detection': return load_data_PV_Attack_type(dataset, n)
    if kind == 'partial_Oable': return load_data_partial(dataset, n)
    raise ValueError("Unsupported PV_attack scenario")

G, P0Data, Class = load_dataset(args.PV_attack, args.dataset, args.N_Scenarios)
labels = np.asarray(Class)

# -----------------------------
# Feature building (timed)
# -----------------------------
N = list(G.nodes); time_step = len(P0Data[0]['TimeSeries_Voltage'])

t_feat0 = time.time()
feature_list = []
for i in tqdm(range(args.N_Scenarios), desc='Building features'):
    x = P0Data[i]['TimeSeries_Voltage']  # shape [T, |V|, 3]
    # per-node concatenate 3-phase voltages across all timesteps -> [|V|, 3T]
    x_new = x.transpose(1, 0, 2).reshape(len(N), time_step * 3).astype(np.float32)
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x_new).astype(np.float32)
    feature_list.append(x_scaled)

data_list = make_graph_data(G, feature_list, Class)  # list of PyG Data objects
feature_build_time = time.time() - t_feat0
print(f"[Timing] Feature building: {feature_build_time:.2f} s")

num_class = len(np.unique(labels))
in_channels = data_list[0].x.size(1)
print(f"in_channels={in_channels}, num_class={num_class}, graphs={len(data_list)}")

# -----------------------------
# 80/20 stratified split on graph labels
# -----------------------------
idx = np.arange(len(data_list))
train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=args.seed, stratify=labels)

train_data = [data_list[i] for i in train_idx]
test_data  = [data_list[i] for i in test_idx]

# -----------------------------
# DataLoaders
# -----------------------------
def make_loader(dataset_split, batch_size, shuffle):
    pin = (device.type == 'cuda')
    nw = max(0, args.num_workers)
    return DataLoader(dataset_split, batch_size=batch_size, shuffle=shuffle,
                      num_workers=nw, pin_memory=pin,
                      persistent_workers=(nw > 0), prefetch_factor=(2 if nw > 0 else None))

train_loader = make_loader(train_data, args.batch_size, shuffle=True)
test_loader  = make_loader(test_data,  args.batch_size, shuffle=False)

# -----------------------------
# Model factory
# -----------------------------
def build_model(model_name: str):
    if model_name in ['GCN', 'SAGE', 'GAT', 'GIN']:
        m = GNN(model_name, in_channels=in_channels,
                hidden_channels=args.hidden_channels, num_classes=num_class)
    elif model_name == 'UniMP':
        m = GraphTransformer(in_channels=in_channels,
                             hidden_channels=args.hidden_channels, num_classes=num_class)
    elif model_name == 'graphormer':
        m = Graphormer(in_channels=in_channels,
                       hidden_channels=args.hidden_channels, num_classes=num_class)
    elif model_name == 'GPS':
        m = GPSModel(in_channels=in_channels,
                     hidden_channels=args.hidden_channels, num_classes=num_class)
    else:
        raise ValueError(f"Model '{model_name}' is not defined.")
    if args.compile:
        try:
            m = torch.compile(m)
            print("[INFO] Compiled model with torch.compile")
        except Exception as e:
            print(f"[WARN] torch.compile failed: {e}")
    return m.to(device)

model = build_model(args.model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
scaler = GradScaler(enabled=(args.amp and device.type == 'cuda'))

# -----------------------------
# Train / Eval helpers (AMP-aware)
# -----------------------------
def train_one_epoch(model, loader, optimizer, criterion, scaler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None and device.type == 'cuda':
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(data.x, data.edge_index, data.batch)
                loss = criterion(out, data.y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * data.num_graphs
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total += data.y.size(0)
    return total_loss / max(1, len(loader.dataset)), correct / max(1, total)

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_targets = [], []
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        total_loss += loss.item() * data.num_graphs
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total += data.y.size(0)
        all_preds.extend(pred.cpu().numpy())
        all_targets.extend(data.y.cpu().numpy())
    avg_loss = total_loss / max(1, len(loader.dataset))
    acc = correct / max(1, total)
    prec = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    rec  = recall_score(all_targets,  all_preds, average='weighted', zero_division=0)
    f1   = f1_score(all_targets,      all_preds, average='weighted', zero_division=0)
    return avg_loss, acc, prec, rec, f1

# -----------------------------
# Training (timed)
# -----------------------------
if device.type == 'cuda': torch.cuda.synchronize()
t_train0 = time.time()

best = dict(acc=0.0, f1=0.0)
for epoch in range(1, args.epochs + 1):
    tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, scaler=scaler)
    te_loss, te_acc, te_prec, te_rec, te_f1 = evaluate(model, test_loader, criterion)
    if te_f1 > best['f1']:
        best.update(acc=te_acc, f1=te_f1)
    if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
        print(f"Epoch {epoch:03d} | "
              f"Train {tr_loss:.4f}/{tr_acc*100:.2f}% | "
              f"Test {te_loss:.4f}/{te_acc*100:.2f}% P{te_prec*100:.2f} R{te_rec*100:.2f} F1{te_f1*100:.2f}")

if device.type == 'cuda': torch.cuda.synchronize()
training_time = time.time() - t_train0

# -----------------------------
# Testing (pure inference timing)
# -----------------------------
model.eval()
if device.type == 'cuda': torch.cuda.synchronize()
t_test0 = time.time()

# total test time
_ = evaluate(model, test_loader, criterion)

if device.type == 'cuda': torch.cuda.synchronize()
testing_time_total = time.time() - t_test0
testing_time_avg_per_batch = testing_time_total / max(1, len(test_loader))

# -----------------------------
# Report
# -----------------------------
print("\n=== 80/20 Split Results ===")
print(f"Best Test Acc: {best['acc']*100:.2f}% | Best Test F1: {best['f1']*100:.2f}%")
print(f"[Timing] Feature building: {feature_build_time:.2f} s")
print(f"[Timing] Training (total): {training_time:.2f} s")
print(f"[Timing] Testing (total):  {testing_time_total:.2f} s | "
      f"Avg per test batch: {testing_time_avg_per_batch:.4f} s")
