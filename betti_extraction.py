
import networkx as nx
import pyflagser
import pickle
import numpy as np

G=nx.read_gml("data/34 Bus/34busEx.gml")
A = nx.to_numpy_array(G)
N=list(G.nodes)
E=list(G.edges)
with open('data/34 Bus/PVanomalydataset.pkl', 'rb') as g:
    P0Data = pickle.load(g)
N_Senario=len(P0Data)

def TimeSeries_Fe_MP(TS_voltage, TS_branchFlow, F_voltage,F_Flow):
    betti_0=[]
    for k in range(len(TS_voltage)):
        fec=[]
        AverageVoltage=[]
        Voltage=TS_voltage[k]
        for y in Voltage:
            AverageVoltage.append(Average(list(y)))
        #AverageVoltage = [i * 100 for i in AverageVoltage]
        BranchFlow=[]
        Branch_Flow=TS_branchFlow[k]
        for j  in range(len(Branch_Flow)):
            BranchFlow.append(Branch_Flow[j][0])

        for p in range(len(F_voltage)):
            Active_node_v=np.where(np.array(AverageVoltage) > F_voltage[p])[0].tolist()
            for q in range(len(F_Flow)):
                #if AverageVoltage[p]> F_voltage[p] and BranchFlow[q]> F_Flow[q]:
                #n_active = np.where(np.array(AverageVoltage) > F_voltage[p])[0].tolist()
                n_active=Active_node_v.copy()
                #print(n_active)
                G = nx.DiGraph()
                G.add_nodes_from(n_active)
                indices = np.where(np.array(BranchFlow) > F_Flow[q])[0].tolist()
                for s in indices:
                    a=int(N.index(E[s][0]))
                    b=int(N.index(E[s][1]))
                    if a in n_active and b in n_active:
                        G.add_edge(a, b)
                    #n_active.append(int(N.index(E[s][0])))
                    #n_active.append(int(N.index(E[s][1])))
                #Active_node=np.unique(n_active)
                #print(G.edges())
                if (len(n_active)==0):
                    fec.append(0)
                else:
                    #b=A[Active_node,:][:,Active_node]
                    Adj = nx.to_numpy_array(G)
                    my_flag=pyflagser.flagser_unweighted(Adj, min_dimension=0, max_dimension=2, directed=False, coeff=2, approximation=None)
                    x = my_flag["betti"]
                    fec.append(x[0])
                n_active.clear()
            Active_node_v.clear()
        betti_0.append(fec)
    return betti_0

def Average(lst):
    return sum(lst) / len(lst)

F_Flow=[30,25,23,20,17,15,10,7,5,2,0]
F_voltage=[1.05,1.04,1.03,1.02,1.01,1.0,0.99,0.98,0.97,0.96]
Betti_0=[]
#N_Senario=len(P0Data)
N_Senario=300
for i in range(N_Senario):
    print("\rProcessing file {} ({}%)".format(i, 100*i//(N_Senario-1)), end='', flush=True)
    TimeSeries_Voltage=P0Data[i]["TimeSeries_Voltage"]
    TimeSeries_Branch_Flow=P0Data[i]["BranchFlow"]
    betti=TimeSeries_Fe_MP(TimeSeries_Voltage,TimeSeries_Branch_Flow, F_voltage,F_Flow)
    #print(betti)
    #Betti=Make_Sum(betti)
    #print(Betti)
    Betti_0.append(betti)
    #Betti_0.append(Betti)

with open('PV_MP_betti0.data', 'wb') as f:
    pickle.dump(Betti_0, f)
