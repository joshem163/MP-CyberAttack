import os
import networkx as nx
import pickle
from sklearn.model_selection import KFold
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from modules import load_data, Average,TimeSeries_Fe_MP,extract_betti,stat,TimeSeries_Fe_singlePH, load_data_PV
from models import TransformerClassifier,reset_weights
N_Senario=300
attack_Type='PV attack' #sensor attack
dataset='34 Bus'
if attack_Type=='PV attack':
    G, P0Data, Class = load_data_PV(dataset, N_Senario)
elif attack_Type=='sensor attack':
    G, P0Data, Class = load_data(dataset, N_Senario)
else:
    print("dataset is not available for this type of scenario")
# G,P0Data,Class=load_data('37 Bus',N_Senario)
F_Flow=[100,90,75,50,25,10,5,0]
#F_voltage=[1.4,1.3,1.2,1.1,1.0,0.99,0.98,0.97,0.96,0.95,0.94,0.92,0.90,0.89,0.88,0.87,0.86,0.85]
F_voltage=[1.4,1.3,1.2,1.1,1.0,0.99,0.98,0.97,0.96,0.95,0.94,0.92,0.90,0.89,0.88,0.87,0.33,0.30]

# #Single Persistence
# Betti_0=[]
# A = nx.to_numpy_array(G)
# for i in range(N_Senario):
#     print("\rProcessing file {} ({}%)".format(i, 100*i//(N_Senario-1)), end='', flush=True)
#     TimeSeries_Voltage=P0Data[i]["TimeSeries_Voltage"]
#     betti=TimeSeries_Fe_singlePH(A,TimeSeries_Voltage, F_voltage)
#     Betti_0.append(betti)

# #Multi Persistence
#Betti_0=extract_betti(G,P0Data,N_Senario,F_voltage, F_Flow)
# from sklearn.preprocessing import MinMaxScaler
## Baseline transformer using node voltage only
A = nx.to_numpy_array(G)
N=list(G.nodes)
E=list(G.edges)
# def voltage_timeseries(scenario, total_time_step):
#     time_series_voltage=[]
#     for time_step in range(total_time_step):
#         voltage=[]
#         for node in range(len(N)):
#             node_voltage=P0Data[scenario][ 'TimeSeries_Voltage'][time_step][node]
#             #voltage.append(P0Data[scenario]['Bus Voltages'][node][time_step].tolist())
#             voltage.append(Average(node_voltage))
#         time_series_voltage.append(voltage)
#     return time_series_voltage
# node_voltage=[]
# for i in range(N_Senario):
#     time_series=voltage_timeseries(i,24)
#     node_voltage.append(time_series)
with open('Features/PV_MP_betti0New.data', 'rb') as f:
    Betti_0 = pickle.load(f)
#X0=node_voltage # for node voltage
X0=Betti_0 # for betti vectorization
y=Class

XX=[]
for i in range(N_Senario):

    scaler = MinMaxScaler()

# Fit scaler to data and transform data
    XX.append(scaler.fit_transform(X0[i]))

# Convert data to PyTorch tensors
X = torch.tensor(XX, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

num_samples = N_Senario
num_timesteps = len(X[0])
num_features = len(X[0][0])
num_classes = len(np.unique(y))

# Define input and output dimensions (example placeholders)
input_dim = num_features
hidden_dim = 64
output_dim = num_classes
n_heads = 2
n_layers = 2
num_timesteps = num_timesteps  # Adjust based on your sequence length

# Initialize model, loss function, and optimizer
model = TransformerClassifier(input_dim, hidden_dim, output_dim, n_heads, n_layers, num_timesteps)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# K-Fold Cross Validation
kfold = KFold(n_splits=10, shuffle=True)
loss_per_fold = []
acc_per_fold = []
pre_per_fold = []
rec_per_fold = []
f1_per_fold = []
fold_no = 1

for train_idx, test_idx in kfold.split(X):
    # Split data
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Create DataLoader
    train_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    # Lists to store metrics
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    # Train the model
    #reset_weights(model)
    model.train()
    for epoch in tqdm(range(100), desc="Processing"):
        epoch_train_loss = 0
        correct_train = 0
        total_train = 0

        # Training loop
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            # Track training loss and accuracy
            epoch_train_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total_train += y_batch.size(0)
            correct_train += (predicted == y_batch).sum().item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_accuracy = correct_train / total_train
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

        # Evaluate on the test set
        model.eval()
        correct_test = 0
        total_test = 0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                output = model(X_batch)
                _, predicted = torch.max(output, 1)
                total_test += y_batch.size(0)
                correct_test += (predicted == y_batch).sum().item()

                # Store predictions and targets for metrics
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())

        test_accuracy = correct_test / total_test
        test_accuracies.append(test_accuracy)

        # Calculate precision, recall, and F1-score
        precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

        model.train()  # Switch back to training mode

        # Print metrics for this epoch
        # print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Train Accuracy = {train_accuracy:.2f}%, Test Accuracy = {test_accuracy:.2f}%')
        # print(f'Precision = {precision:.2f}, Recall = {recall:.2f}, F1-Score = {f1:.2f}')
    # print(f'Score for fold {fold_no}: ')
    # accuracy=print_stat(train_accuracies,test_accuracies)
    accuracy = np.max(test_accuracies)
    pre = np.max(precisions)
    rec = np.max(recalls)
    f1 = np.max(f1_scores)
    acc_per_fold.append(accuracy)
    pre_per_fold.append(pre)
    rec_per_fold.append(rec)
    f1_per_fold.append(f1)

    print(
        f'Score for fold {fold_no}: Test Accuracy = {accuracy:.2f}%, Precision = {pre:.2f}%,recall = {rec:.2f}%, f1 score = {f1:.2f}%')
    #     with open("out_protiens.txt", "w") as file:
    #         with redirect_stdout(file):
    #             print(f'Score for fold {fold_no}: Test Accuracy = {accuracy:.2f}%')
    fold_no += 1
print('Result Statistics acc pre, rec and f1 respectively')
stat(acc_per_fold)
stat(pre_per_fold)
stat(rec_per_fold)
stat(f1_per_fold)