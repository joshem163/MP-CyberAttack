import os
import networkx as nx
import numpy as np
from tqdm import tqdm
import warnings
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from modules import (
    load_data_partial, load_data_PV_multi_attack, load_data_PV,
    Average, extract_betti, stat, TimeSeries_Fe_singlePH
)

warnings.filterwarnings("ignore")

# Define classifier type ('MLP', 'RF', 'XGB')
classifier = 'MLP'  # Change this to try different classifiers
N_Scenarios = 500  # Number of scenarios
PV_attack = 'partial_Oable'  # Attack type,partial_Oable,SinglePV,MultiplePV
dataset = '123 bus'  # Dataset name

# Load dataset based on attack type
if PV_attack == 'SinglePV':
    G, P0Data, Class = load_data_PV(dataset, N_Scenarios)
elif PV_attack == 'MultiplePV':
    G, P0Data, Class = load_data_PV_multi_attack(dataset, N_Scenarios)
elif PV_attack == 'partial_Oable':
    G, P0Data, Class = load_data_partial(dataset, N_Scenarios)
else:
    raise ValueError("Dataset is not available for this type of scenario")

# Convert graph to adjacency matrix
A = nx.to_numpy_array(G)
N = list(G.nodes)
E = list(G.edges)


# Function to compute average voltage over time for each node
def voltage_timeseries(scenario, total_time_step):
    time_series_voltage = np.array([
        [np.mean(P0Data[scenario]['TimeSeries_Voltage'][time_step][node]) for node in range(len(N))]
        for time_step in range(total_time_step)
    ])
    return time_series_voltage


# Compute mean voltage for each scenario
node_voltage = [
    np.sum(voltage_timeseries(i, len(P0Data[1]['TimeSeries_Voltage'])), axis=0)
    for i in tqdm(range(N_Scenarios), desc='Processing scenarios')
]

# Convert to NumPy arrays
X = np.array(node_voltage)
y = np.array(Class)

# Feature scaling (important for MLP and XGBoost)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Define K-fold Cross Validator
num_folds = 10
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Initialize metrics storage
acc_per_fold, precision_per_fold, recall_per_fold, f1_per_fold = [], [], [], []

# Perform K-Fold Cross-Validation
for fold_no, (train, test) in enumerate(kfold.split(X, y), start=1):
    # Select classifier
    if classifier == 'XGB':
        model = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1, booster='gbtree', n_jobs=-1)
    elif classifier == 'MLP':
        model = MLPClassifier(solver='lbfgs', alpha=1e-3, hidden_layer_sizes=(100,50), activation='logistic',
                              learning_rate='constant', learning_rate_init=0.01, random_state=1, max_iter=300)
    elif classifier == 'RF':
        model = RandomForestClassifier(n_estimators=300, max_depth=5, bootstrap=True, n_jobs=-1)
    else:
        raise ValueError('Model does not exist')

    # Train the model
    model.fit(X[train], y[train])

    # Predict on the test set
    y_pred = model.predict(X[test])

    # Compute performance metrics
    acc = accuracy_score(y[test], y_pred) * 100
    precision = precision_score(y[test], y_pred, average="weighted") * 100
    recall = recall_score(y[test], y_pred, average="weighted") * 100
    f1 = f1_score(y[test], y_pred, average="weighted") * 100

    # Store metrics
    acc_per_fold.append(acc)
    precision_per_fold.append(precision)
    recall_per_fold.append(recall)
    f1_per_fold.append(f1)

    # Print fold results
    print(
        f"Fold {fold_no}: Accuracy = {acc:.2f}%, Precision = {precision:.2f}%, Recall = {recall:.2f}%, F1 Score = {f1:.2f}%")

# Compute and display overall performance
print("\nOverall Metrics:")
print(f"Mean Accuracy: {np.mean(acc_per_fold):.2f} ± {np.std(acc_per_fold):.2f}%")
print(f"Mean Precision: {np.mean(precision_per_fold):.2f} ± {np.std(precision_per_fold):.2f}%")
print(f"Mean Recall: {np.mean(recall_per_fold):.2f} ± {np.std(recall_per_fold):.2f}%")
print(f"Mean F1 Score: {np.mean(f1_per_fold):.2f} ± {np.std(f1_per_fold):.2f}%")