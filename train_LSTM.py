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
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import KFold
import warnings
import numpy as np
import tensorflow as tf

from sklearn.metrics import precision_score, recall_score, f1_score
from modules import load_data, Average,TimeSeries_Fe_MP,extract_betti,stat,TimeSeries_Fe_singlePH,load_data_PV
from models import TransformerClassifier,reset_weights
warnings.filterwarnings("ignore")
N_Senario=300

attack_Type='PV attack' #sensor attack
dataset='34 Bus'
if attack_Type=='PV attack':
    G, P0Data, Class = load_data_PV(dataset, N_Senario)
elif attack_Type=='sensor attack':
    G, P0Data, Class = load_data('37 Bus', N_Senario)
else:
    print("dataset is not available for this type of scenario")
num_timesteps=len(P0Data[1]['TimeSeries_Voltage'])
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
    time_series=voltage_timeseries(i,num_timesteps)
    node_voltage.append(time_series)






# Data
X = np.array(node_voltage)
y = np.array(Class)

# Parameters
num_samples = N_Senario
num_timesteps = len(X[0])
num_features = len(X[0][0])
num_classes = 4
num_epochs = 500
batch_size = 32
num_folds = 10
learning_rate=0.001

# Define the K-Fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)

# Metrics storage
loss_per_fold = []
acc_per_fold = []
precision_per_fold = []
recall_per_fold = []
f1_per_fold = []

fold_no = 1


# Custom callback to print test accuracy at each epoch
class TestAccuracyCallback(tf.keras.callbacks.Callback):
    def __init__(self, X_test, y_test):
        super().__init__()
        self.X_test = X_test
        self.y_test = y_test

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 10 == 0:  # Print every 10 epochs
            y_pred = np.argmax(self.model.predict(self.X_test), axis=1)
            test_accuracy = accuracy_score(self.y_test, y_pred)
            print(f"Epoch {epoch + 1},Loss = {logs['loss']:.4f}, Train Accuracy = {logs['accuracy']:.4f} - Test Accuracy: {test_accuracy:.4f}")


for train, test in kfold.split(X, y):
    print(f"\nTraining on Fold {fold_no}...")

    # Define the LSTM model
    model = Sequential([
        LSTM(10, activation='tanh', recurrent_activation='sigmoid', dropout=0.2, input_shape=(num_timesteps, num_features)),
        Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Normalize the features for the entire dataset
    scaler = MinMaxScaler()
    X_scaled = np.array([scaler.fit_transform(sample) for sample in X])

    # Create and apply the custom callback for test accuracy
    test_callback = TestAccuracyCallback(X_scaled[test], y[test])

    # Train the model
    model.fit(
        X_scaled[train], y[train],
        epochs=num_epochs,
        batch_size=batch_size,
        callbacks=[test_callback],
        verbose=0  # Suppress detailed output
    )

    # Evaluate the model
    loss, test_accuracy = model.evaluate(X_scaled[test], y[test], verbose=0)
    y_pred = np.argmax(model.predict(X_scaled[test]), axis=1)

    # Calculate additional metrics
    precision = precision_score(y[test], y_pred, average='weighted')
    recall = recall_score(y[test], y_pred, average='weighted')
    f1 = f1_score(y[test], y_pred, average='weighted')

    print(f"Fold {fold_no} - Test Loss: {loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    # Store metrics
    loss_per_fold.append(loss)
    acc_per_fold.append(test_accuracy * 100)
    precision_per_fold.append(precision)
    recall_per_fold.append(recall)
    f1_per_fold.append(f1)

    fold_no += 1

# Summary of metrics
print("\nCross-Validation Metrics:")
print(f"Mean Loss: {np.mean(loss_per_fold):.4f}\u00B1 {np.std(loss_per_fold):.4f}")
print(f"Mean Accuracy: {np.mean(acc_per_fold):.2f}\u00B1 {np.std(acc_per_fold):.2f}%")
print(f"Mean Precision: {np.mean(precision_per_fold):.4f}\u00B1 {np.std(precision_per_fold):.4f}")
print(f"Mean Recall: {np.mean(recall_per_fold):.4f}\u00B1 {np.std(recall_per_fold):.4f}")
print(f"Mean F1 Score: {np.mean(f1_per_fold):.4f}\u00B1 {np.std(f1_per_fold):.4f}")
