## Model results

This directory contains saved results and models from running the notebooks
in `../modeling/` and analysis of these results. In particular, the results
and saved models are titled as follows $PREFIX_$SUFFIX. The prefixes indicate:

- `RNNCUSTOM`: results/models from fitting RNN-type models without pre-trained
embeddings

- `RNNLaw2Vec`: results/models from fitting RNN-type models with the Law2Vec
pre-trained embeddings

- `SimpleNNCustom`: results/models from fitting Simple NN-type models without
pre-trained embeddings

- `SimpleNNLaw2Vec`: results/models from fitting Simple NN-type models with the
Law2Vec pre-trained embeddings

The suffixes indicate:

- `_best_acc.pt`: saved PyTorch model with the highest accuracy for a given
prefix type

- `_best_prec.pt`: saved PyTorch model with the highest accuracy for a given
prefix type

- `_best_rec.pt`: saved PyTorch model with the highest accuracy for a given
prefix type

- `_models.csv`: saved model results from the best model across all the
different architectures trained for a given prefix type

Additionally, this directory also contains `model_performance_results.ipynb`,
which has analysis of the saved models and results.