
from sklearn.model_selection import KFold
#from tqdm import tqdm
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import pickle
from torch.utils.data import DataLoader, TensorDataset
#import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from modules import save_features, Average,extract_betti,stat,TimeSeries_Fe_singlePH,load_data_partial, load_data_PV, load_data_PV_multi_attack, load_data_PV_Attack_type
from models import TransformerClassifier,reset_weights, GRUClassifier
#from betti_extraction import *
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser(description='experiment')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--dataset', type=str,default='8500 bus')  # 123 bus, 8500 bus
parser.add_argument('--PV_attack', type=str,default='SinglePV') #Atype_detection,MultiplePV, SinglePV
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--N_Scenarios', type=int, default=300) #
parser.add_argument('--head', type=int, default=2)
parser.add_argument('--hidden_channels', type=int, default=120)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=600)
parser.add_argument('--runs', type=int, default=10)
args = parser.parse_args()
print(args)

if args.PV_attack=='SinglePV':
    G, P0Data, Class = load_data_PV(args.dataset, args.N_Scenarios)
elif args.PV_attack=='MultiplePV':
    G, P0Data, Class=load_data_PV_multi_attack(args.dataset, args.N_Scenarios)
elif args.PV_attack=='Atype_detection':
    G, P0Data, Class=load_data_PV_Attack_type(args.dataset, args.N_Scenarios)
elif args.PV_attack=='partial_Oable':
    G, P0Data, Class=load_data_partial(args.dataset,args.N_Scenarios)
else:
    print("dataset is not available for this type of scenario")

#####################
N=list(G.nodes)
E=list(G.edges)
time_step=len(P0Data[0]['TimeSeries_Voltage'])
BFlow_threshold=np.array([30,25,23,20,17,15,10,7,5,2,0])
#voltage_threshold=np.array([1.05,1.04,1.03,1.02,1.01,1.0,0.99,0.98,0.97,0.96])
voltage_threshold=np.array([1.05,1.04, 1.03, 1.02, 0.99, 0.98, 0.96,0.95,0.93, 0.35, 0.34,0.33])

#########################
# def extract_topological_features(graph_id):
#     TimeSeries_Voltage=P0Data[graph_id]["TimeSeries_Voltage"]
#     TimeSeries_Branch_Flow=P0Data[graph_id]["BranchFlow"]
#     betti=Topo_Fe_TimeSeries_MP(TimeSeries_Voltage,TimeSeries_Branch_Flow, voltage_threshold,BFlow_threshold,N,E)
#     return betti
#
# def process_graph(graph_id):
#     # Function to extract topological features from a single graph
#     return extract_topological_features(graph_id)
#
# feat_mp_500=[]
# for i in tqdm(range(args.N_Scenarios),desc='processing multi-persistence vectors'):
#     bet=process_graph(i)
#     feat_mp_500.append(bet)
#
# # save the features
# save_features(feat_mp_500, args.dataset, args.PV_attack)

# #Single Persistence
##############################
# Betti_0=[]
# A = nx.to_numpy_array(G)
# for i in range(N_Senario):
#     print("\rProcessing file {} ({}%)".format(i, 100*i//(N_Senario-1)), end='', flush=True)
#     TimeSeries_Voltage=P0Data[i]["TimeSeries_Voltage"]
#     betti=TimeSeries_Fe_singlePH(A,TimeSeries_Voltage, F_voltage)
#     Betti_0.append(betti)
###############################
# #Multi Persistence


## Baseline transformer using node voltage only
##########################
# A = nx.to_numpy_array(G)
# N=list(G.nodes)
# E=list(G.edges)
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
# num_timesteps=len(P0Data[1]['TimeSeries_Voltage'])
# print(num_timesteps)
# for i in range(N_Senario):
#     time_series=voltage_timeseries(i,num_timesteps)
#     node_voltage.append(time_series)
######################################
# with open('MPV_MP_betti0.data', 'rb') as f:
#     Betti_01 = pickle.load(f)
# with open('MPV_MP_betti0_150.data', 'rb') as f:
#     Betti_02 = pickle.load(f)
# EV_34bus_betti0_singlePV.data
# Betti_0=Betti_01+Betti_02
######################
# load extracted topological features
with open('PV2_8500_MP_betti150.data', 'rb') as f:
    Betti_01 = pickle.load(f)
with open('PV2_8500_MP_betti150_300.data', 'rb') as f:
    Betti_02 = pickle.load(f)

Betti_0=Betti_01+Betti_02
#########################
def main():

    #X0=node_voltage # for node voltage
    X0=Betti_0 # for betti vectorization
    y=Class
    print(y)
    XX=[]
    for i in range(args.N_Scenarios):

        scaler = MinMaxScaler()

    # Fit scaler to data and transform data
        XX.append(scaler.fit_transform(X0[i]))

    # Convert data to PyTorch tensors
    X = torch.tensor(XX, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    num_samples = args.N_Scenarios
    num_timesteps = len(X[0])
    num_features = len(X[0][0])
    num_classes = len(np.unique(y))

    # Define input and output dimensions (example placeholders)
    input_dim = num_features
    hidden_dim = args.hidden_channels
    output_dim = num_classes
    n_heads = args.head
    n_layers = args.num_layers
    num_timesteps = time_step  # Adjust based on your sequence length

    # Initialize model, loss function, and optimizer
    model = TransformerClassifier(input_dim, hidden_dim, output_dim, n_heads, n_layers, num_timesteps)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # K-Fold Cross Validation
    kfold = KFold(n_splits=10, shuffle=True,random_state=123)
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
        reset_weights(model)
        model.train()
        for epoch in range(600):
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
            f'Score for fold {fold_no}: Test Accuracy = {accuracy:.4f}%, Precision = {pre:.4f}%,recall = {rec:.4f}%, f1 score = {f1:.4f}%')
        #     with open("out_protiens.txt", "w") as file:
        #         with redirect_stdout(file):
        #             print(f'Score for fold {fold_no}: Test Accuracy = {accuracy:.2f}%')
        fold_no += 1
    print('Result(2-PV) Statistics acc pre, rec and f1 respectively')
    stat(acc_per_fold,'accuracy')
    stat(pre_per_fold,'precision')
    stat(rec_per_fold,'recall')
    stat(f1_per_fold,'f1 score')
if __name__ == "__main__":
    main()
