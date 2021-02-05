import itertools
import os
import pickle
import random

import numpy as np
import torch
import yaml
from sklearn.cluster import SpectralCoclustering
from sklearn.model_selection import train_test_split
from torch.multiprocessing import Pool
from tqdm import tqdm

from choice_models import fit, MultinomialLogit, MultinomialCDM, Logit, CDM, ConditionalLogit, LCL, \
    ConditionalMultinomialLogit, MultinomialLCL, MultinomialLowRankCDM, MixedLogit, mixed_logit_em
from choice_set_models import ItemIdentityLogisticRegression
from datasets import SFShop, SFWork, Expedia, SyntheticPetsMNAR, SyntheticPetsMCAR, SyntheticPetsCSIgnorableMNAR, \
    SyntheticPetsCSIgnorableMCAR, \
    SyntheticPetsPIgnorableMCAR, SyntheticPetsPIgnorableMNAR, SyntheticPetsNonIgnorableMNAR, \
    SyntheticPetsNonIgnorableMCAR, SyntheticConfoundedCDMSuper, YoochooseCats

with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

DATA_DIR = config['datadir']


L2_LAMBDA = 0.0001
EPOCHS = 500

N_THREADS = 40

RESULTS_DIR = 'results'


def synthtic_pets_ipw_helper(args):
    seed, MNARClass, MCARClass = args

    choice_sets, choices, person_df, propensities = MNARClass.generate_pytorch(5000, seed)
    w = 1 / propensities

    X_train, X_test, chooser_train, chooser_test, y_train, y_test, w_train, w_test = train_test_split(
        choice_sets, torch.from_numpy(person_df.values).float(), choices, w,
        test_size=0.2,
        random_state=0)

    mcar_choice_sets, mcar_choices, mcar_person_df, _ = MCARClass.generate_pytorch(5000, seed+1)
    mcar_person_feats = torch.from_numpy(mcar_person_df.values).float()

    results = dict()

    for use_ipw in [True, False]:
        weights = w_train if use_ipw else torch.ones_like(w_train)
        for ModelClass in [Logit, CDM, MultinomialLogit, MultinomialCDM]:
            is_chooser_model = ModelClass in [MultinomialLogit, MultinomialCDM]

            n_items = 3
            n_chooser_feats = 1

            fit_args = [X_train, chooser_train, y_train, weights] if is_chooser_model else [X_train, y_train, weights]
            test_args = [X_test, chooser_test] if is_chooser_model else [X_test]
            constructor_args = [n_items, n_chooser_feats] if is_chooser_model else [n_items]

            model = ModelClass(*constructor_args)

            fit(model, fit_args, epochs=EPOCHS, learning_rate=0.06, l2_lambda=L2_LAMBDA, show_progress=False)
            pred = model(*test_args)
            loss = torch.nn.functional.nll_loss(pred, y_test).item()
            mrr = model.mean_relative_rank(pred, y_test).item()

            mcar_test_args = [mcar_choice_sets, mcar_person_feats] if is_chooser_model else [mcar_choice_sets]
            mcar_pred = model(*mcar_test_args)

            mcar_loss = torch.nn.functional.nll_loss(mcar_pred, mcar_choices).item()
            mcar_mrr = model.mean_relative_rank(mcar_pred, mcar_choices).item()

            no_bird_set = torch.tensor([[[1], [1], [0]]]).float()
            bird_set = torch.tensor([[[1], [1], [1]]]).float()
            cat_person = torch.tensor([[1]]).float()
            dog_person = torch.tensor([[0]]).float()

            if not is_chooser_model:
                dog_no_bird_est = np.exp(model(no_bird_set).detach().numpy())[0, 1, 0]
                dog_bird_est = np.exp(model(bird_set).detach().numpy())[0, 1, 0]
            else:
                dog_no_bird_est = 0.25 * np.exp(model(no_bird_set, cat_person).detach().numpy())[0, 1, 0] \
                                  + 0.75 * np.exp(model(no_bird_set, dog_person).detach().numpy())[0, 1, 0]
                dog_bird_est = 0.25 * np.exp(model(bird_set, cat_person).detach().numpy())[0, 1, 0] \
                                  + 0.75 * np.exp(model(bird_set, dog_person).detach().numpy())[0, 1, 0]

            results[use_ipw, ModelClass] = loss, mrr, mcar_loss, mcar_mrr, dog_no_bird_est, dog_bird_est

    return args, results


def synthetic_pets_ipw_experiment():
    print('Running synthetic_pets_ipw_experiment()')
    trials = 16

    params = [(seed, MNARClass, MCARClass) for seed in range(trials) for MNARClass, MCARClass in
              ((SyntheticPetsMNAR, SyntheticPetsMCAR),
               (SyntheticPetsCSIgnorableMNAR, SyntheticPetsCSIgnorableMCAR),
               (SyntheticPetsPIgnorableMNAR, SyntheticPetsPIgnorableMCAR),
               (SyntheticPetsNonIgnorableMNAR, SyntheticPetsNonIgnorableMCAR))]

    all_results = []
    with Pool(N_THREADS) as pool:
        for args, results in tqdm(pool.imap(synthtic_pets_ipw_helper, params), total=len(params)):
            all_results.append((args, results))

    with open(f'{RESULTS_DIR}/synthetic_pets_ipw_all.pickle', 'wb') as f:
        pickle.dump(all_results, f)


def item_identity_dataset_helper(args):
    dataset, learning_rate, split_seed, ModelClass, use_ipw, full_dataset = args
    is_chooser_model = ModelClass in [MultinomialLogit, MultinomialCDM, MultinomialLowRankCDM]

    choice_sets, choices, person_df = dataset.load_pytorch()

    chooser_feats = torch.from_numpy(person_df.values).float()

    w = dataset.get_ipw_weights() if use_ipw else torch.ones(len(choices))

    X_train, X_test, chooser_train, chooser_test, y_train, y_test, w_train, w_test = train_test_split(
        choice_sets.float(), chooser_feats, choices, w,
        test_size=0.2,
        random_state=split_seed)

    if full_dataset:
        X_train, chooser_train, y_train, w_train = choice_sets.float(), chooser_feats, choices, w
        y_test = choices

    n_items = choice_sets.size(1)
    n_chooser_feats = chooser_feats.size(1)

    fit_args = [X_train, chooser_train, y_train, w_train] if is_chooser_model else [X_train, y_train, w_train]
    test_args = fit_args[:-2] if full_dataset else ([X_test, chooser_test] if is_chooser_model else [X_test])
    constructor_args = [n_items, n_chooser_feats] if is_chooser_model else [n_items]

    model = ModelClass(*constructor_args)

    loss = fit(model, fit_args, epochs=EPOCHS, learning_rate=learning_rate, l2_lambda=L2_LAMBDA, show_progress=False)
    mrr = model.mean_relative_rank(model(*test_args), y_test).item()
    return args, model.state_dict(), loss, mrr, model.num_params


def item_feature_dataset_helper(args):
    dataset, learning_rate, split_seed, ModelClass, use_ipw, full_dataset = args
    is_chooser_model = ModelClass in [ConditionalMultinomialLogit, MultinomialLCL]

    choice_set_features, choice_set_lengths, choices, person_df = dataset.load_pytorch()
    chooser_feats = torch.from_numpy(person_df.values).float()

    w = torch.ones(len(choices))
    if use_ipw:
        w = dataset.get_ipw_weights()

    X_train, X_test, lengths_train, lengths_test, chooser_train, chooser_test, y_train, y_test, w_train, w_test = train_test_split(
        choice_set_features, choice_set_lengths, chooser_feats, choices, w, test_size=0.2,
        random_state=split_seed)

    if full_dataset:
        X_train, lengths_train, chooser_train, y_train, w_train = choice_set_features, choice_set_lengths, chooser_feats, choices, w
        y_test = choices

    n_item_feats = choice_set_features.size(2)
    n_chooser_feats = chooser_feats.size(1)

    fit_args = [X_train, lengths_train, chooser_train, y_train, w_train] if is_chooser_model else [X_train, lengths_train, y_train, w_train]
    test_args = fit_args[:-2] if full_dataset else ([X_test, lengths_test, chooser_test] if is_chooser_model else [X_test, lengths_test])
    constructor_args = [n_item_feats, n_chooser_feats] if is_chooser_model else [n_item_feats]

    model = ModelClass(*constructor_args)

    loss = fit(model, fit_args, epochs=EPOCHS, learning_rate=learning_rate, l2_lambda=L2_LAMBDA, show_progress=False, show_live_loss=False)
    mrr = model.mean_relative_rank(model(*test_args), y_test).item()
    return args, model.state_dict(), loss, mrr, model.num_params


def run_item_identity_models(datasets, models, use_ipw, full_dataset):
    print(f'Running run_item_identity_models({datasets}, {models}, {use_ipw}, {full_dataset})')
    learning_rates = np.logspace(np.log10(0.001), np.log10(0.1), 3)
    split_seeds = [0]

    params = list(itertools.product(datasets, learning_rates, split_seeds, models, use_ipw, [full_dataset]))

    results = {dataset: [] for dataset in datasets}
    with Pool(N_THREADS) as pool:
        for args, state_dict, loss, mrr, num_params in tqdm(pool.imap_unordered(item_identity_dataset_helper, params), total=len(params)):
            results[args[0]].append((args, state_dict, loss, mrr, num_params))

    for dataset in results:
        fname = f'{RESULTS_DIR}/{dataset.name}_compare_models{"_full_data" if full_dataset else ""}.pt'

        with open(fname, 'wb') as f:
            torch.save(results[dataset], f)


def run_item_feature_models(datasets, models, use_ipw, full_dataset=False):
    print(f'Running run_item_feature_models({datasets}, {models}, {use_ipw}, {full_dataset})')

    learning_rates = np.logspace(np.log10(0.001), np.log10(0.1), 3)
    split_seeds = [0]

    params = list(itertools.product(datasets, learning_rates, split_seeds, models, use_ipw, [full_dataset]))

    results = {dataset: [] for dataset in datasets}
    with Pool(N_THREADS) as pool:
        for args, state_dict, loss, mrr, num_params in tqdm(pool.imap_unordered(item_feature_dataset_helper, params), total=len(params)):
            results[args[0]].append((args, state_dict, loss, mrr, num_params))

    for dataset in results:
        fname = f'{RESULTS_DIR}/{dataset.name}_compare_models{"_full_data" if full_dataset else ""}.pt'

        with open(fname, 'wb') as f:
            torch.save(results[dataset], f)


def synthetic_confounded_cdm_experiment_helper(args):
    ModelClass, learning_rate, use_ipw, samples, embedding_dim, seed, context_strength, confounding_strength = args

    is_chooser_model = ModelClass in [MultinomialLogit, MultinomialCDM, MultinomialLowRankCDM]

    choice_sets, choices, person_df, propensities = SyntheticConfoundedCDMSuper.generate_pytorch(samples, embedding_dim,
                                                                                                 seed, context_strength,
                                                                                                 confounding_strength)

    chooser_feats = torch.from_numpy(person_df.values).float()

    w = torch.ones(len(choices))

    if use_ipw:
        shifted_choice_sets = ((2 * choice_sets) - 1).squeeze().numpy()

        model = ItemIdentityLogisticRegression(SyntheticConfoundedCDMSuper)
        model.train(person_df.values, shifted_choice_sets)

        choice_set_probs = model.choice_set_assignment_probs(person_df.values, shifted_choice_sets)

        w = 1 / (choice_set_probs / np.max(choice_set_probs) + 0.0001)
        w = torch.from_numpy(w).float()

    X_train, X_test, chooser_train, chooser_test, y_train, y_test, w_train, w_test = train_test_split(
        choice_sets.float(), chooser_feats, choices, w,
        test_size=0.2,
        random_state=seed)

    n_items = choice_sets.size(1)
    n_chooser_feats = chooser_feats.size(1)

    fit_args = [X_train, chooser_train, y_train, w_train] if is_chooser_model else [X_train, y_train, w_train]
    test_args = ([X_test, chooser_test] if is_chooser_model else [X_test])
    constructor_args = [n_items, n_chooser_feats] if is_chooser_model else [n_items]

    model = ModelClass(*constructor_args)

    train_loss = fit(model, fit_args, epochs=EPOCHS, learning_rate=learning_rate, l2_lambda=L2_LAMBDA, show_progress=False)
    test_pred = model(*test_args)
    test_loss = torch.nn.functional.nll_loss(test_pred, y_test, reduction='mean').item()
    test_mrr = model.mean_relative_rank(test_pred, y_test).item()

    # MCAR evaluation of model
    mcar_choice_sets, mcar_choices, person_df, mcar_propensities = SyntheticConfoundedCDMSuper.generate_pytorch(samples,
                                                                                                 embedding_dim,
                                                                                                 seed,
                                                                                                 context_strength,
                                                                                                 0)

    mcar_args = [mcar_choice_sets, torch.from_numpy(person_df.values).float()] if is_chooser_model else [mcar_choice_sets]
    mcar_pred = model(*mcar_args)
    mcar_loss = torch.nn.functional.nll_loss(mcar_pred, mcar_choices, reduction='mean').item()
    mcar_mrr = model.mean_relative_rank(mcar_pred, mcar_choices).item()

    return args, model.state_dict(), train_loss, test_loss, test_mrr, model.num_params, mcar_loss, mcar_mrr


def run_synthetic_confounded_cdm_experiment():
    print(f'Running run_synthetic_confounded_cdm_experiment()...')

    models = [Logit, CDM, MultinomialLogit, MultinomialCDM]
    learning_rates = [0.01]
    use_ipws = [True, False]
    samples = [10000]
    embedding_dims = [2]
    seeds = list(range(16))
    context_strengths = [1]

    confounding_strengths = np.linspace(0, 8, 21)

    params = list(itertools.product(models, learning_rates, use_ipws, samples, embedding_dims, seeds,
                                    context_strengths, confounding_strengths))
    random.shuffle(params)

    results = dict()
    with Pool(N_THREADS) as pool:
        for args, state_dict, train_loss, test_loss, test_mrr, num_params, mcar_loss, mcar_mrr in tqdm(
                pool.imap_unordered(synthetic_confounded_cdm_experiment_helper, params), total=len(params)):
            results[args] = state_dict, train_loss, test_loss, test_mrr, num_params, mcar_loss, mcar_mrr

    fname = f'{RESULTS_DIR}/synthetic_counfounded_cdm_results.pt'
    with open(fname, 'wb') as f:
        torch.save(results, f)


def spectral_clustering_experiment_helper(args):
    dataset, lr, seed, ModelClass, n_clusters = args

    np.random.seed(seed)

    choice_sets, choices, _ = dataset.load_pytorch()

    spec_clusters = SpectralCoclustering(n_clusters=n_clusters, random_state=seed).fit(choice_sets.squeeze().numpy()).row_labels_
    rand_clusters = np.random.permutation(spec_clusters)

    spec_results = []
    rand_results = []

    n_items = choice_sets.size(1)

    for cluster in sorted(set(spec_clusters)):
        for clusters, results in zip([spec_clusters, rand_clusters], [spec_results, rand_results]):

            cluster_idx = clusters == cluster
            cluster_choice_sets = choice_sets[cluster_idx]
            cluster_choices = choices[cluster_idx]
            w = torch.ones(len(cluster_choices))

            model = ModelClass(n_items)

            loss = fit(model, (cluster_choice_sets, cluster_choices, w), epochs=EPOCHS, learning_rate=lr,
                       l2_lambda=L2_LAMBDA,
                       show_progress=False)

            results.append((len(cluster_choices), loss, model.state_dict(), model.num_params))

    return args, spec_results, rand_results


def spectral_clustering_experiment(datasets, models):
    print(f'Running spectral_clustering_experiment()')
    learning_rates = np.logspace(np.log10(0.001), np.log10(0.1), 3)
    split_seeds = range(8)
    n_clusters = range(2, 11)

    params = list(itertools.product(datasets, learning_rates, split_seeds, models, n_clusters))

    results = {dataset: [] for dataset in datasets}
    with Pool(N_THREADS) as pool:
        for args, spec_results, rand_results in tqdm(pool.imap_unordered(spectral_clustering_experiment_helper, params),
                                                            total=len(params)):
            results[args[0]].append((args, spec_results, rand_results))

    for dataset in results:
        fname = f'{RESULTS_DIR}/{dataset.name}_spectral_clustering.pt'

        with open(fname, 'wb') as f:
            torch.save(results[dataset], f)


def mixed_logit_em_experiment_helper(args):
    dataset, n_components = args

    choice_sets, choices, _ = dataset.load_pytorch()
    model = MixedLogit(choice_sets.size(1), n_components)

    model, loss = mixed_logit_em(model, choice_sets, choices)

    return args, loss, model.state_dict(), model.num_params


def run_mixed_logit_em_experiment(datasets, n_components):
    print(f'Running run_mixed_logit_em({datasets}, {n_components})')
    params = list(itertools.product(datasets, n_components))

    results = {dataset: [] for dataset in datasets}
    with Pool(N_THREADS) as pool:
        for args, loss, state_dict, num_params in tqdm(pool.imap_unordered(mixed_logit_em_experiment_helper, params),
                                                            total=len(params)):
            results[args[0]].append((args, loss, state_dict, num_params))

    for dataset in results:
        fname = f'{RESULTS_DIR}/{dataset.name}_mixed_logit_em.pt'

        with open(fname, 'wb') as f:
            torch.save(results[dataset], f)


if __name__ == '__main__':
    # Annoying PyTorch bugfix (https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667)
    torch.multiprocessing.set_sharing_strategy('file_system')

    os.makedirs(f'{DATA_DIR}/pickles/', exist_ok=True)
    os.makedirs(f'results/', exist_ok=True)
    os.makedirs(f'ipw-weights/', exist_ok=True)

    # Make pickles
    for dataset in [Expedia, YoochooseCats]:
        dataset.load()

    # Prepare IPW weights
    for dataset in [SFWork, SFShop, Expedia]:
        dataset.get_ipw_weights()

    run_synthetic_confounded_cdm_experiment()
    synthetic_pets_ipw_experiment()

    run_item_identity_models([SFWork, SFShop], [Logit, CDM, MultinomialLogit, MultinomialCDM], use_ipw=[True, False], full_dataset=True)
    run_item_feature_models([Expedia], [ConditionalLogit, LCL, ConditionalMultinomialLogit, MultinomialLCL], use_ipw=[True, False], full_dataset=True)

    spectral_clustering_experiment([YoochooseCats], [Logit])
    run_item_identity_models([YoochooseCats], [Logit], use_ipw=[False], full_dataset=True)
    run_mixed_logit_em_experiment([YoochooseCats], range(2, 11))
