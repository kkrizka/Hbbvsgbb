Analysis code for Hbb vs gluon->bb discrimination.

# Environment Setup
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

# Local Settings
Modify the `settings.py` file to correspond to your workstation.

The available settings are:
- `datadir`: Location of data samples.

# Code Organization
The following modules inside the `hbbgbb` packages are defined for performing common tasks.
- `data`: Loading and decoration of samples.
- `analysis`: Common tasks for analyzing the perfomance of a tagger.
- `plot`: Common plots for tagger performance.
- `formatter`: Automatic formatting of mpl figures.
- `models`: Package with trained models implemented as Sonnet modules.

# Running
The main scripts can be run either as standalone Python or inside a VSCode notebook.

## Training A Model
Several scripts exist to train and test different models for discrimination. Most can be configured via command line arguments (see `--help` for details).

The main output of each is a npy file with the resulting ROC curves. Those can then be compared via the `compare_roc.py` script.

The following models exist:
- `model_atlasxbb`: Discrimination using existin ATLAS Xbb tagger.
- `model_feat`: Trains a NN with large R jet features.
- `model_graph`: Trains a GNN with tracks associatied to a large R jet.

Example:
```shell
python model_feat.py --output simplenn --epochs 100
python model_graph.py --output graphnn --epochs 100
```

## Comparing Results
The output `npy` files from model training scripts can be compared with the `compare_roc` script.

Example:
```shell
python compare_roc.py roc_simplenn.npy roc_graphnn.npy
```