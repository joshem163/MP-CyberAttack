import networkx as nx
import pickle
import numpy as np

G = nx.read_gml("data/8500Bus/8500NCEx.gml")
A = nx.to_numpy_array(G)
N = list(G.nodes)
E = list(G.edges)
with open('data/8500Bus/2-PV.pkl', 'rb') as g:
    P0Data = pickle.load(g)
for item in P0Data:
    item['TimeSeries_Voltage'] = np.nan_to_num(item['TimeSeries_Voltage'], nan=-1)


# N_Senario=len(P0Data)


def Topo_Fe_TimeSeries_MP(TS_voltage, TS_branchFlow, F_voltage, F_Flow):
    betti_0 = []

    for k in range(len(TS_voltage)):
        fec = []
        Voltage = TS_voltage[k]

        # Compute AverageVoltage using NumPy (vectorized operation)
        AverageVoltage = np.array([Average(y) for y in Voltage])

        # Extract first column of TS_branchFlow using NumPy for efficiency
        BranchFlow = np.array([bf[0] for bf in TS_branchFlow[k]])

        for p in range(len(F_voltage)):
            # Precompute active nodes based on threshold F_voltage[p]
            Active_node_v = np.where(AverageVoltage > F_voltage[p])[0]

            for q in range(len(F_Flow)):
                if Active_node_v.size == 0:
                    fec.append(0)
                    continue

                # Create directed graph
                G = nx.Graph()
                G.add_nodes_from(Active_node_v)

                # Find edges where branch flow exceeds threshold F_Flow[q]
                indices = np.where(BranchFlow > F_Flow[q])[0]
                edges_to_add = [(int(N.index(E[s][0])), int(N.index(E[s][1]))) for s in indices]

                # Filter edges to include only active nodes
                edges_to_add = [(a, b) for a, b in edges_to_add if a in Active_node_v and b in Active_node_v]
                G.add_edges_from(edges_to_add)
                fec.append(nx.number_connected_components(G))

        betti_0.append(fec)

    return betti_0


def Average(lst):
    return sum(lst) / len(lst)


F_Flow = [100, 30, 25, 23, 20, 15, 10, 5, 2, 0]
# F_voltage=[1.05,1.04,1.03,1.01,1.0,0.99,0.97,0.96,0.94,0.92,0.88,0.35,0.33,0.30]
F_voltage = [1.05, 1.04, 1.03, 1.0, 0.97, 0.96, 0.94, 0.68, 0.67, 0.66, 0.35, 0.34, 0.33]
Betti_0 = []
# N_Senario=len(P0Data)
N_Senario = 150
for i in range(N_Senario):
    print("\rProcessing file {} ({}%)".format(i, 100 * i // (N_Senario - 1)), end='', flush=True)
    TimeSeries_Voltage = P0Data[i]["TimeSeries_Voltage"]
    TimeSeries_Branch_Flow = P0Data[i]["BranchFlow"]
    betti = Topo_Fe_TimeSeries_MP(TimeSeries_Voltage, TimeSeries_Branch_Flow, F_voltage, F_Flow)
    Betti_0.append(betti)

with open('PV2_8500_MP_betti150.data', 'wb') as f:
    pickle.dump(Betti_0, f)
