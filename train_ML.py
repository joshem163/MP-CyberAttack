import os
import networkx as nx
from sklearn.model_selection import KFold
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
from tqdm import tqdm
import numpy as np

from sklearn.metrics import precision_score, recall_score, f1_score
from modules import load_data, Average,TimeSeries_Fe_MP,extract_betti,stat,TimeSeries_Fe_singlePH,load_data_PV
from models import TransformerClassifier,reset_weights
warnings.filterwarnings("ignore")
classifier = 'XGB'#RF,XGB,MLP
N_Senario=300
attack_Type='PV attack' #sensor attack
dataset='34 Bus'
if attack_Type=='PV attack':
    G, P0Data, Class = load_data_PV(dataset, N_Senario)
elif attack_Type=='sensor attack':
    G, P0Data, Class = load_data(dataset, N_Senario)
else:
    print("dataset is not available for this type of scenario")
timestep=len(P0Data[1]['TimeSeries_Voltage'])
# G,P0Data,Class=load_data('123 Bus',N_Senario)
A = nx.to_numpy_array(G)
N=list(G.nodes)
E=list(G.edges)
def Average(lst):
    return sum(lst) / len(lst)
def voltage_timeseries(scenario, total_time_step):
    time_series_voltage=[]
    for time_step in range(total_time_step):
        voltage=[]
        for node in range(len(N)):
            node_voltage=P0Data[scenario][ 'TimeSeries_Voltage'][time_step][node]
            #voltage.append(P0Data[scenario]['Bus Voltages'][node][time_step].tolist())
            voltage.append(Average(node_voltage))
        time_series_voltage.append(voltage)
    return time_series_voltage
node_voltage=[]
for i in range(N_Senario):
    time_series=voltage_timeseries(i,timestep)
    mean_volatge=np.sum(np.array(time_series), axis=0)
    node_voltage.append(mean_volatge)



# Convert data to NumPy arrays (if not already)
X = np.array(node_voltage)
y = np.array(Class)

# Define the K-fold Cross Validator
num_folds = 10
kfold = KFold(n_splits=num_folds, shuffle=True)

# Initialize variables for tracking metrics
acc_per_fold = []
precision_per_fold = []
recall_per_fold = []
f1_per_fold = []

# Perform K-Fold Cross-Validation
for fold_no, (train, test) in enumerate(kfold.split(X, y), start=1):
    # Initialize and train the classifier
    if classifier == 'XGB':
        model = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, booster='gbtree')
    elif classifier == 'MLP':
        model = MLPClassifier(solver='lbfgs', alpha=1e-3, hidden_layer_sizes=(100,), activation='logistic',
                              learning_rate='constant', learning_rate_init=0.01, random_state=1, max_iter=500,
                              warm_start=True)
    elif classifier == 'RF':
        model = RandomForestClassifier(n_estimators=200, max_depth=5, bootstrap=True)
    else:
        print('model does not exists')
    model.fit(X[train], y[train])
    # Predict and evaluate metrics
    y_pred = model.predict(X[test])

    acc = accuracy_score(y[test], y_pred) * 100
    precision = precision_score(y[test], y_pred, average="weighted") * 100
    recall = recall_score(y[test], y_pred, average="weighted") * 100
    f1 = f1_score(y[test], y_pred, average="weighted") * 100

    # Store metrics
    acc_per_fold.append(acc)
    precision_per_fold.append(precision)
    recall_per_fold.append(recall)
    f1_per_fold.append(f1)

    # Print metrics for the current fold
    print(
        f"Fold {fold_no}: Accuracy = {acc:.2f}%, Precision = {precision:.2f}%, Recall = {recall:.2f}%, F1 Score = {f1:.2f}%")

# Calculate and print overall metrics
mean_acc = np.mean(acc_per_fold)
std_acc = np.std(acc_per_fold)
mean_precision = np.mean(precision_per_fold)
mean_recall = np.mean(recall_per_fold)
mean_f1 = np.mean(f1_per_fold)

print("\nOverall Metrics:")
print(f"Mean Accuracy: {mean_acc:.2f} \u00B1 {std_acc:.2f}%")
print(f"Mean Precision: {mean_precision:.2f} \u00B1 {np.std(precision_per_fold):.2f}%")
print(f"Mean Recall: {mean_recall:.2f} \u00B1 {np.std(recall_per_fold):.2f}%")
print(f"Mean F1 Score: {mean_f1:.2f} \u00B1 {np.std(f1_per_fold):.2f}%")
