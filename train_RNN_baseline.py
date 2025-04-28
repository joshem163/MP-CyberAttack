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
from modules import Average, stat, load_data_PV, load_data_PV_multi_attack, load_data_PV_Attack_type, load_data_partial
from models import reset_weights, LSTMClassifier, RNNClassifier, GRUClassifier

# Define parameters for dataset and model selection
N_Scenarios = 300  # Number of scenarios to consider
dataset = '123 bus'  # Dataset identifier
model_type = 'GRU'  # Model type selection (RNN, GRU, LSTM)
PV_attack = 'partial_Oable'  # Attack type scenario selection

# Load dataset based on attack type
if PV_attack == 'SinglePV':
    G, P0Data, Class = load_data_PV(dataset, N_Scenarios)
elif PV_attack == 'MultiplePV':
    G, P0Data, Class = load_data_PV_multi_attack(dataset, N_Scenarios)
elif PV_attack == 'Atype_detection':
    G, P0Data, Class = load_data_PV_Attack_type(dataset, N_Scenarios)
elif PV_attack == 'partial_Oable':
    G, P0Data, Class = load_data_partial(dataset, N_Scenarios)
else:
    print("Dataset is not available for this type of scenario")

# Determine the number of timesteps from dataset
timestep = len(P0Data[1]['TimeSeries_Voltage'])

# Convert network graph to adjacency matrix
A = nx.to_numpy_array(G)
N = list(G.nodes)
print(len(N))  # Print number of nodes in the graph


# Function to extract time series voltage data
def voltage_timeseries(scenario, total_time_step):
    time_series_voltage = []
    for time_step in range(total_time_step):
        voltage = []
        for node in range(len(N)):
            node_voltage = P0Data[scenario]['TimeSeries_Voltage'][time_step][node]
            voltage.append(Average(node_voltage))  # Compute average voltage per node
        time_series_voltage.append(voltage)
    return time_series_voltage


# Process all scenarios to extract voltage time series data
node_voltage = []
num_timesteps = len(P0Data[1]['TimeSeries_Voltage'])
for i in tqdm(range(N_Scenarios), desc='Processing'):
    time_series = voltage_timeseries(i, num_timesteps)
    node_voltage.append(time_series)

# Define feature matrix (X) and labels (y)
X0 = node_voltage  # Node voltage time series data
y = Class  # Corresponding labels

# Normalize the data using MinMaxScaler
XX = []
for i in range(N_Scenarios):
    scaler = MinMaxScaler()
    XX.append(scaler.fit_transform(X0[i]))

# Convert data to PyTorch tensors
X = torch.tensor(np.array(XX), dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)
print(y)  # Print labels

# Define model hyperparameters
num_samples = N_Scenarios
num_features = len(X[0][0])
num_classes = len(np.unique(y))
num_timesteps = len(X[0])
input_dim = num_features
hidden_dim = 64
output_dim = num_classes
num_layers = 2
print('Results using', model_type, 'model')

# Initialize the selected model
if model_type == 'RNN':
    model = RNNClassifier(input_dim, hidden_dim, output_dim, num_timesteps, num_layers)
elif model_type == 'LSTM':
    model = LSTMClassifier(input_dim, hidden_dim, output_dim, num_timesteps, num_layers)
elif model_type == 'GRU':
    model = GRUClassifier(input_dim, hidden_dim, output_dim, num_timesteps, num_layers)
else:
    print('This model has not been used!')

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# K-Fold Cross Validation Setup
kfold = KFold(n_splits=10, shuffle=True)
loss_per_fold = []
acc_per_fold = []
pre_per_fold = []
rec_per_fold = []
f1_per_fold = []
fold_no = 1

# Perform k-fold cross-validation
for train_idx, test_idx in kfold.split(X):
    # Split data into training and testing sets
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    train_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    # Initialize tracking variables for metrics
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    # Reset model weights before training
    reset_weights(model)
    model.train()

    # Train the model
    for epoch in tqdm(range(400), desc=f"Fold {fold_no}"):
        epoch_train_loss = 0
        correct_train = 0
        total_train = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total_train += y_batch.size(0)
            correct_train += (predicted == y_batch).sum().item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_accuracy = correct_train / total_train
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

        # Evaluate the model
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

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())

        test_accuracy = correct_test / total_test
        test_accuracies.append(test_accuracy)

        precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

        model.train()

    # Store best metrics for each fold
    accuracy = np.max(test_accuracies)
    pre = np.max(precisions)
    rec = np.max(recalls)
    f1 = np.max(f1_scores)
    acc_per_fold.append(accuracy)
    pre_per_fold.append(pre)
    rec_per_fold.append(rec)
    f1_per_fold.append(f1)

    print(
        f'Score for fold {fold_no}: Test Accuracy = {accuracy:.4f}%, Precision = {pre:.4f}%,recall = {rec:.4f}%, f1 score = {f1:.4f}%')
    fold_no += 1

# Display final statistics
print('Result Statistics: Accuracy, Precision, Recall, F1 Score')
stat(acc_per_fold, 'accuracy')
stat(pre_per_fold, 'precision')
stat(rec_per_fold, 'recall')
stat(f1_per_fold, 'f1 score')