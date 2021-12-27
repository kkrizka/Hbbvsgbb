# %%
#%load_ext autoreload
#%autoreload 2

# %%
import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
import graph_nets as gn
import sonnet as snt
import matplotlib.pyplot as plt

import hbbgbb.plot as myplt
from hbbgbb import data
from hbbgbb import analysis

import settings

# %% Formatting
from hbbgbb import formatter
fmt=formatter.Formatter('variables.yaml')

# %% Load per jet information
df_train=data.load_data()
data.label(df_train)

df_test=data.load_data('r9364')
data.label(df_test)

# %% Load jet constituent data
feat=['pt','deta','dphi','trk_btagIp_d0','trk_btagIp_z0SinTheta','trk_btagIp_d0Uncertainty','trk_btagIp_z0SinThetaUncertainty']

fjc_train=data.load_data_constit()
g_train=data.create_graphs(df_train, fjc_train,feat)

fjc_test=data.load_data_constit('r9364')
g_test=data.create_graphs(df_test, fjc_test,feat)

#%%
l_train=tf.convert_to_tensor(df_train[['label0','label1','label2']])
l_test =tf.convert_to_tensor(df_test [['label0','label1','label2']])

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
        self.stat = pd.DataFrame(columns=['train_loss','test_loss'])
        self.opt  = snt.optimizers.Adam(learning_rate=0.1)

    def step(self, graphs, labels, g_test=None, l_test=None):
        """Performs one optimizer step on a single mini-batch."""
        # Write test data
        test_loss=0.
        if g_test is not None:
            pred = self.model(g_test)
            logits=pred.globals
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                            labels=l_test)
            test_loss = tf.reduce_mean(loss)

        # Training
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
        self.stat=self.stat.append({'train_loss':float(loss), 'test_loss':float(test_loss)}, ignore_index=True)

        return loss

# %% Training
t = Trainer(graph_network)

fig_s,ax_s=plt.subplots(ncols=3,figsize=(24,8))
fig_t,ax_t=plt.subplots(figsize=(8,8))

epochs=1000
for epoch in tqdm.trange(epochs):
    loss=float(t.step(g_train,l_train, g_test, l_test))
    print(epoch, loss)

    pred=t.model(g_test)
    df_test['pred']=tf.argmax(pred.globals, axis=1)
    predsm=tf.nn.softmax(pred.globals)
    df_test['score0']=predsm[:,0]
    df_test['score1']=predsm[:,1]
    df_test['score2']=predsm[:,2]

    for i in range(3):
        ax_s[i].clear()
        myplt.labels(df_test,f'score{i}','label',fmt=fmt, ax=ax_s[i])
        ax_s[i].set_yscale('log')
    fig_s.savefig('score')

    ax_t.clear()
    ax_t.plot(t.stat.train_loss,label='Training')
    ax_t.plot(t.stat.test_loss,label='Test')
    ax_t.set_yscale('log')
    ax_t.set_ylabel('loss')
    ax_t.set_ylim(1e-1, 1e3)
    ax_t.set_xlabel('epoch')
    ax_t.legend()
    fig_t.savefig('training')
