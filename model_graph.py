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

import hbbgbb.plot as myplt
from hbbgbb import data
from hbbgbb import analysis

import settings

# %% Formatting
from hbbgbb import formatter
fmt=formatter.Formatter('variables.yaml')

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
    track_feat=['trk_btagIp_d0','trk_btagIp_z0SinTheta','trk_btagIp_d0Uncertainty','trk_btagIp_z0SinThetaUncertainty']
    nodes=const[track_feat].to_numpy()

    # Fully connected graph, w/o loops
    i=itertools.product(range(nodes.shape[0]),range(nodes.shape[0]))
    senders=[]
    receivers=[]
    for s,r in i:
        if s==r: continue
        senders.append(s)
        receivers.append(r)
    edges=[[]]*len(senders)

    return {'globals':globals, 'nodes':nodes, 'edges':edges, 'senders':senders, 'receivers':receivers}

mydf=df.copy()

dgraphs=[]
for i,fatjet in tqdm.tqdm(mydf.iterrows(),total=len(mydf.index)):
    fatjet_const=pd.DataFrame(fatjet_consts[i])
    fatjet_const=fatjet_const[~fatjet_const.pt.isna()]
    dgraphs.append(create_graph(fatjet, fatjet_const))

graphs=gn.utils_tf.data_dicts_to_graphs_tuple(dgraphs)
#%%
labels=tf.convert_to_tensor(mydf[['label0','label1','label2']])

# %%
class NormedMLP(snt.Module):
    def __init__(self,output_sizes):
        super(NormedMLP, self).__init__()
        self.norm = snt.LayerNorm(-1, create_scale=True, create_offset=True)
        self.mlp = snt.nets.MLP(output_sizes)
    def __call__(self, input):
        output = self.norm(input)
        return self.mlp(output)

graph_network_0 = gn.modules.InteractionNetwork(
    node_model_fn=lambda: NormedMLP([64]),
    edge_model_fn=lambda: NormedMLP([64]),
)

graph_network_1 = gn.modules.InteractionNetwork(
    node_model_fn=lambda: snt.nets.MLP([64]),
    edge_model_fn=lambda: snt.nets.MLP([64])
)

graph_network_2 = gn.modules.InteractionNetwork(
    node_model_fn=lambda: snt.nets.MLP([64]),
    edge_model_fn=lambda: snt.nets.MLP([64])
)
        
graph_network_out = gn.modules.DeepSets(
    node_model_fn=lambda: snt.nets.MLP([64]),
    global_model_fn=lambda: snt.nets.MLP([64,2])
)

graph_network_out = gn.modules.RelationNetwork(
    edge_model_fn=lambda: NormedMLP([10]),
    global_model_fn=lambda: snt.nets.MLP([3])
)

graph_network = graph_network_out #snt.Sequential([graph_network_0,graph_network_out])

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
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
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

fig_s,ax_s=plt.subplots(ncols=3,figsize=(24,8))
fig_t,ax_t=plt.subplots(figsize=(8,8))

epochs=20
for epoch in range(epochs):
    loss=float(t.step(graphs,labels))
    print(epoch, loss)

    pred=t.model(graphs)
    mydf['pred']=tf.argmax(pred.globals, axis=1)
    predsm=tf.nn.softmax(pred.globals)
    mydf['score0']=predsm[:,0]
    mydf['score1']=predsm[:,1]
    mydf['score2']=predsm[:,2]

    for i in range(3):
        myplt.labels(mydf,f'score{i}','label',fmt=fmt, ax=ax_s[i])
    fig_s.savefig(f'score_{epoch:08d}')
    fig_s.clf()

    ax_t.plot(t.stat.loss)
    ax_t.set_yscale('log')
    ax_t.set_ylabel('loss')
    #plt.ylim(1e-1, 1e2)
    ax_t.set_xlabel('epoch')
    fig_t.savefig('training')
    fig_t.clf()

# %%
plt.plot(t.stat.loss)
plt.yscale('log')
plt.ylabel('loss')
#plt.ylim(1e-1, 1e2)
plt.xlabel('epoch')
plt.savefig('training.png')
plt.show()
plt.clf()

# %%
pred=graph_network(graphs)
mydf['pred']=tf.argmax(pred.globals, axis=1)
predsm=tf.nn.softmax(pred.globals)
mydf['score0']=predsm[:,0]
mydf['score1']=predsm[:,1]
mydf['score2']=predsm[:,2]

# %%
myplt.labels(mydf,'score0','label',fmt=fmt)
plt.savefig('score0')
plt.show()
plt.clf()
# %%
myplt.labels(mydf,'score1','label',fmt=fmt)
plt.savefig('score1')
plt.show()
plt.clf()
# %%
myplt.labels(mydf,'score2','label',fmt=fmt)
plt.savefig('score2')
plt.show()
plt.clf()

# %%
output='graph'
analysis.roc(mydf, 'score0', f'roc_{output}')


# %%
