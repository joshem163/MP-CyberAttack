import os, time, random, warnings, argparse
import numpy as np
from tqdm import tqdm
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score

from modules import Average, stat, load_data_PV, load_data_PV_multi_attack, load_data_PV_Attack_type, load_data_partial
from models import reset_weights, LSTMClassifier, RNNClassifier, GRUClassifier

warnings.filterwarnings("ignore", category=UserWarning)

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def load_dataset(pv_attack, dataset, n_scenarios):
    if pv_attack == 'SinglePV':   return load_data_PV(dataset, n_scenarios)
    if pv_attack == 'MultiplePV': return load_data_PV_multi_attack(dataset, n_scenarios)
    if pv_attack == 'Atype_detection': return load_data_PV_Attack_type(dataset, n_scenarios)
    if pv_attack == 'partial_Oable':   return load_data_partial(dataset, n_scenarios)
    raise ValueError(f"Unsupported PV_attack: {pv_attack}")

def build_features(G, P0Data, n_scenarios):
    N = list(G.nodes)
    T = len(P0Data[0]['TimeSeries_Voltage'])
    def ts_for(i):
        ts=[]
        for t in range(T):
            v=[Average(P0Data[i]['TimeSeries_Voltage'][t][n]) for n in range(len(N))]
            ts.append(v)
        return ts
    raw=[]
    for i in tqdm(range(n_scenarios), desc='Extracting time series'):
        raw.append(ts_for(i))
    XX=[]
    for i in range(n_scenarios):
        scaler=MinMaxScaler()
        XX.append(scaler.fit_transform(raw[i]).astype(np.float32))  # [T, |V|]
    return torch.tensor(np.array(XX), dtype=torch.float32)          # [Nsc, T, |V|]

def make_model(kind, input_dim, hidden_dim, output_dim, T, layers):
    if kind=='RNN':  return RNNClassifier(input_dim, hidden_dim, output_dim, T, layers)
    if kind=='LSTM': return LSTMClassifier(input_dim, hidden_dim, output_dim, T, layers)
    if kind=='GRU':  return GRUClassifier(input_dim, hidden_dim, output_dim, T, layers)
    raise ValueError(f"Unknown model_type: {kind}")

def train_one_epoch(model, loader, opt, crit, device, use_amp=False, scaler=None):
    model.train()
    tot_loss=0; correct=0; total=0
    for Xb,yb in loader:
        Xb=Xb.to(device, non_blocking=True); yb=yb.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        if use_amp and scaler and device.type=='cuda':
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                out=model(Xb); loss=crit(out,yb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else:
            out=model(Xb); loss=crit(out,yb)
            loss.backward(); opt.step()
        tot_loss += loss.item()*yb.size(0)
        pred = out.argmax(1); correct += (pred==yb).sum().item(); total += yb.size(0)
    return tot_loss/max(1,total), correct/max(1,total)

@torch.no_grad()
def evaluate(model, loader, crit, device, use_amp=False):
    model.eval()
    tot_loss=0; correct=0; total=0
    preds=[]; tgts=[]
    for Xb,yb in loader:
        Xb=Xb.to(device, non_blocking=True); yb=yb.to(device, non_blocking=True)
        if use_amp and device.type=='cuda':
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                out=model(Xb); loss=crit(out,yb)
        else:
            out=model(Xb); loss=crit(out,yb)
        tot_loss += loss.item()*yb.size(0)
        p=out.argmax(1); correct += (p==yb).sum().item(); total += yb.size(0)
        preds.extend(p.cpu().numpy()); tgts.extend(yb.cpu().numpy())
    from sklearn.metrics import precision_score, recall_score, f1_score
    return (tot_loss/max(1,total),
            correct/max(1,total),
            precision_score(tgts,preds,average='weighted',zero_division=0),
            recall_score(tgts,preds,average='weighted',zero_division=0),
            f1_score(tgts,preds,average='weighted',zero_division=0))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='8500 bus')
    ap.add_argument('--pv_attack', default='partial_Oable',
                    choices=['SinglePV','MultiplePV','Atype_detection','partial_Oable'])
    ap.add_argument('--model', default='LSTM', choices=['RNN','LSTM','GRU'])
    ap.add_argument('--n_scenarios', type=int, default=100)
    ap.add_argument('--test_size', type=float, default=0.2)
    ap.add_argument('--epochs', type=int, default=1000)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--amp', action='store_true')
    args = ap.parse_args()
    print(args)

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda':
        torch.backends.cuda.matmul.allow_tf32=True
        torch.backends.cudnn.allow_tf32=True
        try: torch.set_float32_matmul_precision('high')
        except Exception: pass
    print(f"Using device: {device}")

    # Load + features
    G,P0Data,Class = load_dataset(args.pv_attack, args.dataset, args.n_scenarios)
    y = torch.tensor(np.array(Class), dtype=torch.long)
    time_fe=time.time()
    X = build_features(G, P0Data, args.n_scenarios)  # [Nsc, T, |V|]
    fe_time=time.time()-time_fe



    T, F = X.shape[1], X.shape[2]
    C = len(np.unique(Class))
    print(f"X: {X.shape} | classes: {C}")

    # 80/20 split (stratified)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=args.test_size,
                                              random_state=args.seed, stratify=y)

    train_loader = DataLoader(TensorDataset(X_tr,y_tr), batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=(device.type=='cuda'),
                              persistent_workers=(args.workers>0))
    test_loader  = DataLoader(TensorDataset(X_te,y_te), batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=(device.type=='cuda'),
                              persistent_workers=(args.workers>0))

    model = make_model(args.model, input_dim=F, hidden_dim=args.hidden, output_dim=C,
                       T=T, layers=args.layers).to(device)
    reset_weights(model)
    crit = nn.CrossEntropyLoss()
    opt  = optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type=='cuda'))

    # Train (time)
    if device.type=='cuda': torch.cuda.synchronize()
    t0=time.time()
    best_f1=0.0
    for ep in tqdm(range(1, args.epochs+1), desc="Training"):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, crit, device,
                                          use_amp=args.amp, scaler=scaler)
        te_loss, te_acc, te_p, te_r, te_f1 = evaluate(model, test_loader, crit, device, use_amp=args.amp)
        best_f1 = max(best_f1, te_f1)
        if ep==1 or ep%10==0 or ep==args.epochs:
            print(f"Epoch {ep:03d} | tr {tr_loss:.4f}/{tr_acc*100:.2f}% | "
                  f"te {te_loss:.4f}/{te_acc*100:.2f}% P{te_p*100:.2f} R{te_r*100:.2f} F1{te_f1*100:.2f}")
    if device.type=='cuda': torch.cuda.synchronize()
    train_time = time.time()-t0

    # Pure inference timing
    model.eval()
    if device.type=='cuda': torch.cuda.synchronize()
    t1=time.time()
    te_loss, te_acc, te_p, te_r, te_f1 = evaluate(model, test_loader, crit, device, use_amp=args.amp)
    if device.type=='cuda': torch.cuda.synchronize()
    test_time = time.time()-t1

    print("\n=== Final Test (80/20 split) ===")
    print(f"Acc {te_acc*100:.2f}% | P {te_p*100:.2f}% | R {te_r*100:.2f}% | F1 {te_f1*100:.2f}%")
    print(f"Train time (total): {train_time:.5f}s | Test time (total): {test_time:.5f}s")
    print(f"Avg per test batch: {test_time/max(1,len(test_loader)):.6f}s")
    print(f"Feature build: {fe_time:.6f}s")

if __name__ == "__main__":
    main()
