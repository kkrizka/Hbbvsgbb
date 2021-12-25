# %%
#%load_ext autoreload
#%autoreload 2

#%%
import sys

import tensorflow as tf
import sonnet as snt
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

import hbbgbb.plot as myplt
from hbbgbb import data
from hbbgbb import analysis

from hbbgbb.models import SimpleModel

# %% Arguments
features=['mass','C2','D2','e3','Tau21_wta','Tau32_wta','Split12','Split23']
output='feat'
epochs=10
if sys.argv[0]!='ipykernel_launcher': # running in a notebook
    import argparse
    parser = argparse.ArgumentParser(description='Train NN from features')
    parser.add_argument('features', nargs='*', default=features, help='Features to train on.')
    parser.add_argument('--output', type=str, default=output, help='Output name.')
    parser.add_argument('--epochs', type=int, default=epochs, help='Number of epochs to train.')
    args = parser.parse_args()

    features = args.features
    output = args.output
    epochs = args.epochs

# %% Formatting
from hbbgbb import formatter
fmt=formatter.Formatter('variables.yaml')

# %% Load the datset
df=data.load_data()
data.label(df)

# %% Create tensors of features
feat=tf.convert_to_tensor(df[features])
labels=tf.convert_to_tensor(df.label)

# %% Create features
for feature in features+['nConstituents']:
  myplt.labels(df, feature, 'label', fmt=fmt)
  plt.savefig(f'labels_{feature}.png')
  plt.show()
  plt.clf()

# %%
mlp=SimpleModel.SimpleModel()

# %%
opt = snt.optimizers.SGD(learning_rate=0.1)

def step(feat,labels):
  """Performs one optimizer step on a single mini-batch."""
  with tf.GradientTape() as tape:
    logits = mlp(feat, is_training=True)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                          labels=labels)
    loss = tf.reduce_mean(loss)

  params = mlp.trainable_variables
  grads = tape.gradient(loss, params)
  opt.apply(grads, params)
  return loss

# %% Training
df_stat=pd.DataFrame(columns=['epoch','loss'])
for epoch in range(epochs):
    loss=float(step(feat,labels))
    print(epoch, loss)
    df_stat=df_stat.append({'epoch':epoch,'loss':float(loss)}, ignore_index=True)

# %%
plt.plot(df_stat.epoch, df_stat.loss)
plt.yscale('log')
plt.ylabel('loss')
plt.ylim(1e-1, 1e1)
plt.xlabel('epoch')
plt.savefig('training.png')
plt.show()
plt.clf()
# %%
pred=mlp(feat)
df['pred']=tf.argmax(pred, axis=1)
predsm=tf.nn.softmax(pred)
df['score0']=predsm[:,0]
df['score1']=predsm[:,1]
df['score2']=predsm[:,2]
# %% Plot distributions of the two predictions
for feature in features+['nConstituents']:
  myplt.labels(df, feature, 'label', 'pred', fmt=fmt)
  plt.savefig(f'predictions_{feature}.png')
  plt.show()
  plt.clf()

# %%
myplt.labels(df,'score0','label',fmt=fmt)
plt.savefig('score0.png')
plt.show()
plt.clf()
# %%
myplt.labels(df,'score1','label',fmt=fmt)
plt.savefig('score1.png')
plt.show()
plt.clf()
# %%
myplt.labels(df,'score2','label',fmt=fmt)
plt.savefig('score2.png')
plt.show()
plt.clf()

# %% Calculate ROC curves
analysis.roc(df, 'score0', f'roc_{output}')

# %%
