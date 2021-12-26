# %%
%load_ext autoreload
%autoreload 2

# %%
import glob

import itertools

import numpy as np
import pandas as pd
import networkx as nx
import graph_nets as gn
import sonnet as snt
import h5py
import matplotlib.pyplot as plt

from hbbgbb import data

import settings

# %% Load per jet information
df=data.load_data()
data.label(df)

# %% Load jet constituent data
path=glob.glob(f'{settings.datadir}/user.zhicaiz.309450.NNLOPS_nnlo_30_ggH125_bb_kt200.hbbTrain.e6281_s3126_r10201_p4258.2020_ftag5dev.v0_output.h5/*.output.h5')[0]
f=h5py.File(path)
fatjet_consts=f['fat_jet_constituents']

# %%
def create_graph(fatjet,const):
    G=nx.Graph()

    # Global features are properties of the fat jet
    G.graph['features']=[]

    # Nodes are individual tracks
    track_feat=['trk_d0','trk_z0','phi','eta']
    nodes=[(idx, {'features':feat.to_numpy()}) for idx, feat in const[track_feat].iterrows()]
    print(nodes)
    G.add_nodes_from(nodes)

    # Fully connected graph, no self-loops
    fc=itertools.product(range(len(const.index)),range(len(const.index)))
    edges=[(i,j,{'features':None}) for i,j in fc if i!=j]
    G.add_edges_from(edges)

    return G

nxgraphs=[]
for i,fatjet in df.iterrows():
    fatjet_const=pd.DataFrame(fatjet_consts[i])
    fatjet_const=fatjet_const[~fatjet_const.pt.isna()]
    nxgraphs.append(create_graph(fatjet, fatjet_const))
    break

graphs=[gn.utils_np.networkx_to_data_dict(g) for g in nxgraphs]
graphs=gn.utils_tf.data_dicts_to_graphs_tuple(graphs)

# %% Draw an example graph
fig, ax = plt.subplots(figsize=(6, 6))
nx.draw(nxgraphs[0], ax=ax)

# %%
OUTPUT_EDGE_SIZE = 10
OUTPUT_NODE_SIZE = 11
OUTPUT_GLOBAL_SIZE = 12
graph_network = gn.modules.DeepSets(
    node_model_fn=lambda: snt.Linear(output_size=OUTPUT_NODE_SIZE),
    global_model_fn=lambda: snt.Linear(output_size=OUTPUT_GLOBAL_SIZE))
# %%
graph_network(graphs)
# %%

# %%
