import networkx as nx
import numpy as np
import pandas as pd
import math
import pyflagser
import pickle
import os
import numpy as np
import statistics
from tqdm import tqdm

graph_path_template = "data/8500Bus/{datasetname} graph.gml"
file_path_template = "data/8500Bus/{PV_attack}_MultipleAttacks.pkl"
file_path_partial = "data/8500Bus/PartialObservdataSPMV123.pkl"
file_path_PV = "data/8500Bus/2-PV.pkl"

# graph_path_template = "/Users/joshem/PhD Research/Power Distribution Network/Anomaly Detection/PV attack/data/{datasetname}/{datasetname} graph.gml"
# file_path_template = "/Users/joshem/PhD Research/Power Distribution Network/Anomaly Detection/PV attack/data/{datasetname}/{PV_attack}_MultipleAttacks.pkl"
# file_path_partial = "/Users/joshem/PhD Research/Power Distribution Network/Anomaly Detection/PV attack/partially_obv/{datasetname}/PartialObservdataSPMV.pkl"
def load_data_PV(dataset_name,N_Senario):
    print(dataset_name, ' dataset is loading.........\n')
    graph_path =graph_path_template.format(datasetname=dataset_name)
    file_path =file_path_PV
   # Read the dataset
    G = nx.read_gml(graph_path)
    with open(file_path, 'rb') as g:
        P0Data = pickle.load(g)
    Class = []
    for i in range(N_Senario):
        Class.append(P0Data[i]["Targeted_PV"])
    class_mapping = {
        'Nil': 0,
        'PV 1': 1,
        'PV 2': 2,
        'PV 3': 3,
        'PV 4': 4,
        'PV 5': 5,
        'PV 6': 6,
        'PV 7': 7,
    }
    for i in range(N_Senario):
        Class[i] = class_mapping[Class[i]]
    print(dataset_name,' dataset has been successfully loaded')
    return G,P0Data,Class


def load_data_partial(dataset_name,N_Senario):
    print(dataset_name, ' dataset is loading.........\n')
    graph_path =graph_path_template.format(datasetname=dataset_name)
    file_path =file_path_partial
   # Read the dataset
    G = nx.read_gml(graph_path)
    with open(file_path, 'rb') as g:
        P0Data = pickle.load(g)
    for item in P0Data:
        item['TimeSeries_Voltage'] = np.nan_to_num(item['TimeSeries_Voltage'], nan=0)
    Class = []
    for i in range(N_Senario):
        Class.append(P0Data[i]["Targeted_PV"])
    class_mapping = {
        'Nil': 0,
        'PV 1': 1,
        'PV 2': 2,
        'PV 3': 3,
        'PV 4': 4,
        'PV 5': 5,
        'PV 6': 6,
        'PV 7': 7,
    }
    for i in range(N_Senario):
        Class[i] = class_mapping[Class[i]]
    print(dataset_name,' partial Observable dataset has been successfully loaded')
    return G,P0Data,Class

def load_data_PV_multi_attack(dataset_name,N_Senario):
    print(dataset_name, ' dataset is loading.........\n')
    graph_path =graph_path_template.format(datasetname=dataset_name)
    file_path =file_path_template.format(PV_attack='MultiplePV')
     # Read the dataset
    G = nx.read_gml(graph_path)
    with open(file_path, 'rb') as g:
        P0Data = pickle.load(g)
    Class = []
    for i in range(N_Senario):
        Class.append(P0Data[i]["Targeted_PV"])
    class_mapping = {
        'Nil': 0,
        'PV 1': 1,
        'PV 2': 2,
        'PV 3': 3,
        'PV 1-3': 4,
        'PV 3 and 1': 5,
        'PV 1 and 3': 5,
        'PV 3 and 2': 6,
        'PV 2 and 3': 6,
        'PV 1 and 2': 7,
        'PV 2 and 1': 7
    }
    for i in range(N_Senario):
        Class[i] = class_mapping[Class[i]]
    print(dataset_name,' dataset has been successfully loaded')
    return G,P0Data,Class

def load_data_PV_Attack_type(dataset_name,N_Senario):
    print(dataset_name, ' dataset is loading.........\n')
    graph_path =graph_path_template.format(datasetname=dataset_name)
    file_path =file_path_template.format(datasetname=dataset_name,PV_attack='SinglePV')
    # Read the dataset
    G = nx.read_gml(graph_path)
    with open(file_path, 'rb') as g:
        P0Data = pickle.load(g)
    Class = []
    for i in range(N_Senario):
        Class.append(P0Data[i]["Attack_Type"])
    # Mapping of class labels to values
    class_mapping = {
        'Nil': 0,
        'DOS': 1,
        'PF': 2,
        'PCA': 3,
    }
    for i in range(N_Senario):
        Class[i] = class_mapping[Class[i]]
    print(dataset_name,' dataset has been successfully loaded')
    return G,P0Data,Class

def Average(lst):
    return sum(lst) / len(lst)
def TimeSeries_Fe_MP(G,node_filtration, edge_filtration, node_thresholds,edge_thresholds):
    betti_0=[]
    for k in range(len(node_filtration)):
        fec=[]
        AverageVoltage=[]
        Voltage=node_filtration[k]
        for y in Voltage:
            AverageVoltage.append(Average(list(y)))
        #AverageVoltage = [i * 100 for i in AverageVoltage]
        BranchFlow=[]
        Branch_Flow=edge_filtration[k]
        for j  in range(len(Branch_Flow)):
            BranchFlow.append(Branch_Flow[j][0])

        for p in range(len(node_thresholds)):
            Active_node_v=np.where(np.array(AverageVoltage) > node_thresholds[p])[0].tolist()
            for q in range(len(edge_thresholds)):
                #if AverageVoltage[p]> F_voltage[p] and BranchFlow[q]> F_Flow[q]:
                #n_active = np.where(np.array(AverageVoltage) > F_voltage[p])[0].tolist()
                n_active=Active_node_v.copy()
                #print(n_active)
                N = list(G.nodes)
                E = list(G.edges)
                subG = nx.DiGraph() # depends on directed or undirected graph
                subG.add_nodes_from(n_active)
                indices = np.where(np.array(BranchFlow) > edge_thresholds[q])[0].tolist()
                for s in indices:
                    a=int(N.index(E[s][0]))
                    b=int(N.index(E[s][1]))
                    if a in n_active and b in n_active:
                        subG.add_edge(a, b) # edge between only active nodes
                if (len(n_active)==0):
                    fec.append(0)
                else:
                    #b=A[Active_node,:][:,Active_node]
                    Adj = nx.to_numpy_array(subG)
                    my_flag=pyflagser.flagser_unweighted(Adj, min_dimension=0, max_dimension=2, directed=False, coeff=2, approximation=None)
                    x = my_flag["betti"]
                    fec.append(x[0])
                n_active.clear()
            Active_node_v.clear()
        betti_0.append(fec)
    return betti_0

def stat(acc_list,metric):
    mean = statistics.mean(acc_list)
    stdev = statistics.stdev(acc_list)
    print('Final',metric, f'using 10 fold CV: {mean*100:.4f} \u00B1 {stdev*100:.4f}%')

def extract_betti(G,P0Data,N_Senario,F_voltage, F_Flow):
    Betti_0 = []
    print('extracting multi-persistence vectors.......')
    for i in tqdm(range(N_Senario), desc="Processing"):
        # print("\rProcessing file {} ({}%)".format(i, 100*i//(N_Senario-1)), end='', flush=True)
        TimeSeries_Voltage = P0Data[i]["TimeSeries_Voltage"]
        TimeSeries_Branch_Flow = P0Data[i]["BranchFlow"]
        betti = TimeSeries_Fe_MP(G,TimeSeries_Voltage, TimeSeries_Branch_Flow, F_voltage, F_Flow)
        # print(betti)
        # Betti=Make_Sum(betti)
        # print(Betti)
        Betti_0.append(betti)
        # Betti_0.append(Betti)
    return Betti_0

def TimeSeries_Fe_singlePH(A,TS_array, F_voltage):
    betti_0=[]
    for k in range(len(TS_array)):
        fec=[]
        AverageVoltage=[]
        Voltage=TS_array[k]
        for y in Voltage:
            AverageVoltage.append(Average(list(y)))
        #AverageVoltage = [i * 10 for i in AverageVoltage]
        for p in range(len(F_voltage)):
            n_active = np.where(np.array(AverageVoltage) > F_voltage[p])[0].tolist()
            Active_node=np.unique(n_active)
            if (len(Active_node)==0):
                fec.append(0)
            else:
                b=A[Active_node,:][:,Active_node]
                my_flag=pyflagser.flagser_unweighted(b, min_dimension=0, max_dimension=2, directed=False, coeff=2, approximation=None)
                x = my_flag["betti"]
                fec.append(x[0])
            n_active.clear()
        betti_0.append(fec)
    return betti_0

def save_features(features, dataset_name, attack_type):
    filename = f"{dataset_name}_{attack_type}.data"
    with open(filename, "wb") as f:
        pickle.dump(features, f)
    print(f"Features saved as {filename}")

