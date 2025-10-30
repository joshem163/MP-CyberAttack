"""
Train Transformer model for graph-based topological time-series classification
with 80/20 train-test split and GPU acceleration.
"""

import argparse
import numpy as np
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
import warnings

from joblib import Parallel, delayed
import multiprocessing

# Local modules
from modules import save_features, Average, stat
from modules import save_features, Average,extract_betti,stat,TimeSeries_Fe_singlePH,load_data_partial, load_data_PV, load_data_PV_multi_attack, load_data_PV_Attack_type,Topo_Fe_TimeSeries_MP
from models import TransformerClassifier,reset_weights, GRUClassifier
from betti_extraction import *

# Ignore warnings for clean logs
warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------- Argument Parser -------------------------
parser = argparse.ArgumentParser(description="Transformer for PV Attack Classification")
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--dataset', type=str, default='8500 bus')  # 123 bus, 8500 bus
parser.add_argument('--PV_attack', type=str, default='partial_Oable')  # Atype_detection, MultiplePV, SinglePV
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--N_Scenarios', type=int, default=100)
parser.add_argument('--head', type=int, default=2)
parser.add_argument('--hidden_channels', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=32)
args = parser.parse_args()

# ------------------------- Device Configuration -------------------------
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
print(f"\nUsing device: {device}")
torch.backends.cudnn.benchmark = True

# ------------------------- Load Dataset -------------------------
if args.PV_attack == 'SinglePV':
    G, P0Data, Class = load_data_PV(args.dataset, args.N_Scenarios)
elif args.PV_attack == 'MultiplePV':
    G, P0Data, Class = load_data_PV_multi_attack(args.dataset, args.N_Scenarios)
elif args.PV_attack == 'Atype_detection':
    G, P0Data, Class = load_data_PV_Attack_type(args.dataset, args.N_Scenarios)
elif args.PV_attack == 'partial_Oable':
    G, P0Data, Class = load_data_partial(args.dataset, args.N_Scenarios)
else:
    raise ValueError("Dataset type not available for this PV_attack")

N=list(G.nodes)
E=list(G.edges)
time_step=len(P0Data[0]['TimeSeries_Voltage'])
BFlow_threshold=np.array([30,25,23,20,17,15,10,7,5,2,0])
#voltage_threshold=np.array([1.05,1.04,1.03,1.02,1.01,1.0,0.99,0.98,0.97,0.96])
voltage_threshold=np.array([1.05,1.04, 1.03, 1.02, 0.99, 0.98, 0.96,0.95,0.93, 0.35, 0.34,0.33])

def extract_topological_features(graph_id):
    TimeSeries_Voltage = P0Data[graph_id]["TimeSeries_Voltage"]
    TimeSeries_Branch_Flow = P0Data[graph_id]["BranchFlow"]
    betti = Topo_Fe_TimeSeries_MP(TimeSeries_Voltage, TimeSeries_Branch_Flow,
                                  voltage_threshold, BFlow_threshold,N,E)
    return betti

num_cores = min(8, multiprocessing.cpu_count())  # cap at 32 cores
#print(f"Using {num_cores} CPU cores for topological feature extraction")

#feat_mp_500 = Parallel(n_jobs=num_cores)(
#    delayed(extract_topological_features)(i) for i in tqdm(range(args.N_Scenarios))
#)
#feat_mp_500 = Parallel(n_jobs=8, backend="threading")(
 #   delayed(Topo_Fe_TimeSeries_MP)(P0Data[i]["TimeSeries_Voltage"],
  #      P0Data[i]["BranchFlow"],
   #     voltage_threshold,
    #    BFlow_threshold,N,E)
   # for i in tqdm(range(args.N_Scenarios))
#)

#save_features(feat_mp_500, args.dataset, args.PV_attack)
#print("Feature extraction completed.")

# Load precomputed topological features
with open('8500_bus_partial_Oable.data', 'rb') as f:
    Betti_0 = pickle.load(f)

# ------------------------- Main Function -------------------------
def main():
    X0 = Betti_0  # Topological vectors
    y = np.array(Class)

    # Normalize per sample
    XX = []
    for i in range(args.N_Scenarios):
        scaler = MinMaxScaler()
        XX.append(scaler.fit_transform(X0[i]))

    X = torch.tensor(XX, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    num_samples, num_timesteps, num_features = X.shape
    num_classes = len(torch.unique(y))
    print(f"\nData loaded: {num_samples} samples, {num_timesteps} timesteps, {num_features} features, {num_classes} classes")

    # 80/20 Train-Test Split
    train_idx, test_idx = train_test_split(
        np.arange(num_samples),
        test_size=0.1,
        random_state=42,
        stratify=y
    )
    print(train_idx)
    print(test_idx)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Create DataLoaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=args.batch_size, shuffle=False)

    # ------------------------- Model Setup -------------------------
    model = TransformerClassifier(
        input_dim=num_features,
        hidden_dim=args.hidden_channels,
        output_dim=num_classes,
        n_heads=args.head,
        n_layers=args.num_layers,
        num_timesteps=num_timesteps
    ).to(device)

    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # ------------------------- Training -------------------------
    print("\n🚀 Starting Training...")
    start_train = time.time()
    model.train()
    train_losses, train_accs = [], []

    for epoch in tqdm(range(args.epochs), desc="Training Epochs"):
        epoch_loss, correct, total = 0.0, 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, pred = torch.max(out, 1)
            correct += (pred == y_batch).sum().item()
            total += y_batch.size(0)

        acc = correct / total
        train_losses.append(epoch_loss / len(train_loader))
        train_accs.append(acc)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}] | Loss: {epoch_loss/len(train_loader):.4f} | Train Acc: {acc*100:.2f}%")

    end_train = time.time()
    train_time = end_train - start_train
    print(f"\n✅ Training completed in {train_time:.2f} seconds")

    # ------------------------- Testing -------------------------
    print("\n🧪 Starting Testing...")
    start_test = time.time()
    model.eval()

    correct, total = 0, 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            out = model(X_batch)
            _, pred = torch.max(out, 1)
            total += y_batch.size(0)
            correct += (pred == y_batch).sum().item()
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
        print(all_preds)
        print(all_targets)

    test_acc = correct / total
    precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)

    end_test = time.time()
    test_time = end_test - start_test

    # ------------------------- Results -------------------------
    print("\n=== 🧾 Performance Summary ===")
    print(f"Train Time: {train_time:.2f} s | Test Time: {test_time:.2f} s")
    print(f"Final Train Accuracy: {train_accs[-1]*100:.2f}%")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
    print(f"Total Runtime: {(train_time + test_time)/60:.2f} min")

    # Optional GPU memory usage info
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        print(f"GPU Memory — Allocated: {allocated:.1f} MB | Reserved: {reserved:.1f} MB")

if __name__ == "__main__":
    main()
