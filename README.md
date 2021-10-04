# Choice Set Confounding in Discrete Choice

This repository accompanies the paper

> Kiran Tomlinson, Johan Ugander, and Austin R. Benson. Choice Set Confounding in Discrete Choice. KDD 2021. 
> [[ACM DL]](https://dl.acm.org/doi/10.1145/3447548.3467378)
> [[arXiv]](https://arxiv.org/abs/2105.07959)


## Libraries
We used:
- Python 3.8.5
  - matplotlib 3.3.0
  - numpy 1.19.1
  - pandas 1.0.5
  - scipy 1.5.1
  - tqdm 4.48.0
  - torch 1.6.0
  - scikit-learn 0.23.1
  
## Files
- `choice_models.py`: implementations of choice models and training
- `choice_set_models.py`: implementations of choice set models and training
- `datasets.py`: dataset processing and management
- `experiments.py`: parallelized model training
- `plot.py`: makes plots and tables in the paper
- `config.yml`: configures location of `data/` directory

The `ipw-weights/` directory contains cached IPW weights in PyTorch format, the `results/` directory contains other experiment outputs, 
and the `data/` directory has the SF datasets.

## Data
We provide the sf-work and sf-shop datasets in `data/` since they are small and not available online. The YOOCHOOSE dataset can be downloaded from https://2015.recsyschallenge.com/challenge.html. Unzip and place the `yoochoose-data/` directory in `data/`. The Expedia dataset can be downloaded from https://www.kaggle.com/c/expedia-personalized-sort/data. Place the `expedia-personalized-sort/` directory in `data/` and uncompress `train.csv`.

## Reproducibility
To create the plots and tables in the paper from the saved results provided in the repository, just run `python3 plot.py`. To rerun all experiments, 
run `python3 experiments.py` after downloading the datasets as described above. By default, `experiments.py` uses 40 threads--you may wish to modify this at the top of the file.
