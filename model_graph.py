# %%
%load_ext autoreload
%autoreload 2

# %%
import glob
import time
import itertools
import tqdm

import numpy as np
import pandas as pd
import networkx as nx
import tensorflow as tf
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
    # Global features are properties of the fat jet
    globals=[]

    # Nodes are individual tracks
    track_feat=['trk_d0','trk_z0','phi','eta']
    nodes=const[track_feat].to_numpy()

    return {'globals':globals, 'nodes':nodes}

graphs=[]
for i,fatjet in tqdm.tqdm(df.iterrows(),total=len(df.index)):
    fatjet_const=pd.DataFrame(fatjet_consts[i])
    fatjet_const=fatjet_const[~fatjet_const.pt.isna()]
    graphs.append(create_graph(fatjet, fatjet_const))

graphs=gn.utils_tf.data_dicts_to_graphs_tuple(graphs)

labels=tf.convert_to_tensor(df[['label']])

# %%
graph_network = gn.modules.DeepSets(
    node_model_fn=lambda: snt.nets.MLP([64, 64, 64]),
    global_model_fn=lambda: snt.nets.MLP([64,3])
)

# %% Training procedure
class Trainer:
    def __init__(self, model):
        # Model to keep track of
        self.model= model

        # Training tools
        self.stat = pd.DataFrame(columns=['loss'])
        self.opt  = snt.optimizers.Adam(learning_rate=0.1)

    def step(self, graphs, labels):
        """Performs one optimizer step on a single mini-batch."""
        with tf.GradientTape() as tape:
            pred = self.model(graphs)
            logits=pred.globals
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                    labels=labels)
            loss = tf.reduce_mean(loss)

        params = self.model.trainable_variables
        grads = tape.gradient(loss, params)
        self.opt.apply(grads, params)

        # save training status
        self.stat=self.stat.append({'loss':float(loss)}, ignore_index=True)

        return loss

# %% Training
t = Trainer(graph_network)

epochs=10
for epoch in range(epochs):
    loss=float(t.step(graphs,labels))
    print(epoch, loss)


# %%
