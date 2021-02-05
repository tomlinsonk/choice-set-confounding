import os
import pickle
from itertools import combinations

import pickle
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import chi2, stats

from choice_models import Logit, CDM, MultinomialCDM, MultinomialLogit, ConditionalLogit, ConditionalMultinomialLogit, \
    MultinomialLCL, LCL
from datasets import Expedia, SFShop, SFWork, ItemIdentityDataset, SyntheticPetsMNAR, SyntheticPetsCSIgnorableMNAR, \
    SyntheticPetsPIgnorableMNAR, \
    SyntheticPetsNonIgnorableMNAR, YoochooseCats
from experiments import RESULTS_DIR


def test_regularity(dataset):
    choice_sets, choices, person_df = dataset.load()

    unique_choice_sets, idx = np.unique(choice_sets, axis=0, return_inverse=True)

    tests = 0
    for i, j in combinations(range(len(unique_choice_sets)), 2):
        c1, c2 = unique_choice_sets[i], unique_choice_sets[j]
        if all(c1 * c2 == c2):
            c1, c2 = c2, c1
            i, j = j, i

        if any(c1 * c2 != c1):
            continue

        c1_counts = np.bincount(choices[idx == i, 0], minlength=len(c1))
        c2_counts = np.bincount(choices[idx == j, 0], minlength=len(c1))
        c1_tot = np.sum(c1_counts)
        c2_tot = np.sum(c2_counts)
        c1_prop = c1_counts / c1_tot
        c2_prop = c2_counts / c2_tot

        c1_names = [dataset.item_names[k] for k in range(len(c1)) if c1[k] == 1]
        c2_names = [dataset.item_names[k] for k in range(len(c1)) if c2[k] == 1]

        for k in range(len(c1)):
            if c1[k] == 1:
                tests += 1
                oddsratio, pvalue = stats.fisher_exact(
                    [[c1_counts[k], c1_tot - c1_counts[k]], [c2_counts[k], c2_tot - c2_counts[k]]])

                if c1_prop[k] < c2_prop[k] and pvalue < 0.05:
                    print(f'{dataset.item_names[k]} (p={pvalue:.2g}])\n'
                          f'\t{c1_prop[k]:.2f} ({c1_tot} samples) in {c1_names} \n'
                          f'\t{c2_prop[k]:.2f} ({c2_tot} samples) in {c2_names} ')


def compare_likelihoods(datasets):
    import warnings
    from torch.serialization import SourceChangeWarning
    warnings.filterwarnings('ignore', category=SourceChangeWarning)

    for col, dataset in enumerate(datasets):
        print(dataset.name)
        results = torch.load(f'{RESULTS_DIR}/{dataset.name}_compare_models_full_data.pt')

        best_losses = dict()
        models = set()
        best_num_params = dict()

        for args, state_dict, loss, mrr, num_params in results:
            dataset, learning_rate, split_seed, model, use_ipw = args[:5]

            if use_ipw:
                continue

            models.add(model)

            if model not in best_losses or loss < best_losses[model]:
                best_losses[model] = loss
                best_num_params[model] = num_params

        is_identity = issubclass(dataset, ItemIdentityDataset)

        if is_identity:
            losses = np.array([[best_losses[Logit], best_losses[MultinomialLogit]],
                               [best_losses[CDM], best_losses[MultinomialCDM]]])
            num_params = np.array([[best_num_params[Logit], best_num_params[MultinomialLogit]],
                                   [best_num_params[CDM], best_num_params[MultinomialCDM]]])
            models = np.array([[Logit, MultinomialLogit], [CDM, MultinomialCDM]])
        else:
            losses = np.array([[best_losses[ConditionalLogit], best_losses[ConditionalMultinomialLogit]],
                               [best_losses[LCL], best_losses[MultinomialLCL]]])
            num_params = np.array([[best_num_params[ConditionalLogit], best_num_params[ConditionalMultinomialLogit]],
                                   [best_num_params[LCL], best_num_params[MultinomialLCL]]])

            models = np.array([[ConditionalLogit, ConditionalMultinomialLogit], [LCL, MultinomialLCL]])

        config = [
            (0.7, (0, 0), (0, 1), None, '#5ebf72', 'covariates', '---'),
            (0, (0, 0), (1, 0), 'Context Effects', '#a3469d', 'context', '---'),
            (0, (1, 0), (1, 1), 'Covariates', '#5ebf72', 'covariates', 'context'),
            (0.7, (0, 1), (1, 1), None, '#a3469d', 'context', 'covariates')
        ]

        losses = -losses

        for y, top, bottom, label, color, testing, controlling in config:
            p = chi2.sf(2 * (losses[bottom] - losses[top]), num_params[bottom] - num_params[top])
            p = f'{p:.2g}' if p > 10 ** -10 else '$< 10^{-10}$'

            print(f'{models[top].table_name} to {models[bottom].table_name} & {testing} & {controlling} & {int(losses[bottom] - losses[top])} & {p}\\\\')


def plot_synthetic_pets_ipw():
    with open(f'{RESULTS_DIR}/synthetic_pets_ipw_all.pickle', 'rb') as f:
        results = pickle.load(f)

    models = [CDM, MultinomialCDM]

    for mcar in [True, False]:
        for MNARClass in (
        SyntheticPetsMNAR, SyntheticPetsCSIgnorableMNAR, SyntheticPetsPIgnorableMNAR, SyntheticPetsNonIgnorableMNAR):

            losses = [[], []]
            ipw_losses = [[], []]

            mrrs = [[], []]
            ipw_mrrs = [[], []]

            dog_bird_ests = [[], []]
            ipw_dog_bird_ests = [[], []]

            dog_no_bird_ests = [[], []]
            ipw_dog_no_bird_ests = [[], []]

            for args, data in results:
                seed, data_class, _ = args
                if data_class != MNARClass:
                    continue

                for ipw in [True, False]:
                    for i in range(2):
                        loss, mrr, mcar_loss, mcar_mrr, dog_no_bird_est, dog_bird_est = data[ipw, models[i]]
                        if ipw:
                            ipw_losses[i].append(mcar_loss if mcar else loss)
                            ipw_mrrs[i].append(mcar_mrr if mcar else mrr)
                            ipw_dog_bird_ests[i].append(dog_bird_est)
                            ipw_dog_no_bird_ests[i].append(dog_no_bird_est)
                        else:
                            losses[i].append(mcar_loss if mcar else loss)
                            mrrs[i].append(mcar_mrr if mcar else mrr)
                            dog_bird_ests[i].append(dog_bird_est)
                            dog_no_bird_ests[i].append(dog_no_bird_est)

            class_names = {SyntheticPetsMNAR: 'Ignorability', SyntheticPetsCSIgnorableMNAR: 'Choice Set Ignorability',
                           SyntheticPetsPIgnorableMNAR: 'Preference Ignorability',
                           SyntheticPetsNonIgnorableMNAR: 'No Ignorability'}

            if mcar:
                plt.figure(figsize=(2.45, 2))
                plt.errorbar(np.arange(2) - 0.1, np.mean(dog_no_bird_ests, axis=1),
                             yerr=np.std(dog_no_bird_ests, axis=1), fmt='o', label='No IPW', capsize=4)
                plt.errorbar(np.arange(2) + 0.1, np.mean(ipw_dog_no_bird_ests, axis=1),
                             yerr=np.std(ipw_dog_no_bird_ests, axis=1), fmt='D', label='IPW', capsize=4)
                plt.hlines(5 / 8, -0.5, 1.5, colors='black', ls='dashed')
                if MNARClass == SyntheticPetsMNAR:
                    plt.legend(loc='lower right')
                plt.xticks(range(2), ['CDM', 'MCDM'])
                plt.ylabel(r'$\mathrm{E}_a[\hat \Pr(\mathrm{dog} \mid \{\mathrm{cat}, \mathrm{dog}\}, a)]$')
                plt.title(f'{class_names[MNARClass]}')
                plt.ylim(0.45, 0.7)

                plt.savefig(f'plots/synthetic-pets-pr-est-{class_names[MNARClass].lower().replace(" ", "-")}.pdf',
                            bbox_inches='tight')

                # plt.show()
                plt.close()

            if MNARClass == SyntheticPetsMNAR:
                plt.figure(figsize=(2, 1.6))
                plt.errorbar(np.arange(2) - 0.1, np.mean(losses, axis=1), yerr=np.std(losses, axis=1), fmt='o',
                             label='No IPW', capsize=4)
                plt.errorbar(np.arange(2) + 0.1, np.mean(ipw_losses, axis=1), yerr=np.std(ipw_losses, axis=1), fmt='D',
                             label='IPW', capsize=4)
                if not mcar:
                    plt.legend(loc='upper right')
                plt.xticks(range(2), ['CDM',  'MCDM'])
                plt.ylabel('Loss')
                plt.title(f'{"Counterfactual Data" if mcar else "Chooser-Dependent Sets"}')
                plt.ylim(0.45, 0.7)
                plt.savefig('plots/synthetic-pets-mcar-loss.pdf' if mcar else 'plots/synthetic-pets-mnar-loss.pdf',
                            bbox_inches='tight')

                # plt.show()
                plt.close()
            #
            #
            # plt.figure(figsize=(3, 2))
            # plt.errorbar(np.arange(4)-0.1, np.mean(mrrs, axis=1), yerr=np.std(mrrs, axis=1), fmt='o', label='No IPW', capsize=4)
            # plt.errorbar(np.arange(4)+0.1, np.mean(ipw_mrrs, axis=1), yerr=np.std(ipw_mrrs, axis=1), fmt='D',  label='IPW', capsize=4)
            # # plt.legend(loc='lower right')
            # plt.xticks(range(4), ['Logit', 'CDM', 'MNL', 'MCDM'])
            # plt.ylabel('Mean Relative Rank')
            # plt.title(f'{"MCAR" if mcar else "MNAR"} MRR {MNARClass}')
            # plt.show()
            #
            # # plt.savefig('synthetic-pets-mcar-mrr.pdf' if mcar else 'synthetic-pets-mnar-mrr.pdf', bbox_inches='tight')
            # plt.close()


def plot_synthetic_confounded_cdm():
    with open(f'{RESULTS_DIR}/synthetic_counfounded_cdm_results.pt', 'rb') as f:
        results = torch.load(f)

    models = [MultinomialCDM, CDM, MultinomialLogit, Logit]
    learning_rates = [0.01]
    use_ipws = [True, False]
    samples = 10000
    n_items = 20
    embedding_dim = 2
    seeds = list(range(16))
    context_strength = 1
    # confounding_strengths = np.concatenate([np.linspace(0, 4, 21), np.linspace(5, 10, 6)])
    confounding_strengths = np.linspace(0, 8, 21)

    linestyles = ['-', '--', '-.', ':']
    colors = ["#65afa2", "#8b57b6", "#a0a745", "#b45255"]

    fig, axes = plt.subplots(1, 2, figsize=(6, 3), sharey='row')

    for i, model in enumerate(models):
        best_results = [[min(
            [results[model, lr, False, samples, n_items, embedding_dim, seed, context_strength, confounding_strength]
             for lr in learning_rates],
            key=lambda x: x[1])
                         for seed in seeds]
                        for confounding_strength in confounding_strengths]

        ipw_best_results = [[min(
            [results[model, lr, True, samples, n_items, embedding_dim, seed, context_strength, confounding_strength]
             for lr in learning_rates],
            key=lambda x: x[1])
                             for seed in seeds]
                            for confounding_strength in confounding_strengths]

        # y: [0: state_dict, 1: train_loss, 2: test_loss, 3: test_mrr, 4: num_params, 5: mcar_loss, 6: mcar_mrr]
        # Plot 1 - mrr (so higher is better)
        mrrs = 1 - np.array([[y[3] for y in x] for x in best_results])
        ipw_mrrs = 1 - np.array([[y[3] for y in x] for x in ipw_best_results])
        mcar_mrrs = 1 - np.array([[y[6] for y in x] for x in best_results])
        ipw_mcar_mrrs = 1 - np.array([[y[6] for y in x] for x in ipw_best_results])

        # mrrs = np.array([[y[2] for y in x] for x in best_results])
        # ipw_mrrs = np.array([[y[2] for y in x] for x in ipw_best_results])
        # mcar_mrrs = np.array([[y[5] for y in x] for x in best_results])
        # ipw_mcar_mrrs = np.array([[y[5] for y in x] for x in ipw_best_results])

        mrr_mean = np.mean(mrrs, axis=1)
        mrr_err = stats.sem(mrrs, axis=1)
        axes[0].plot(confounding_strengths, mrr_mean, label=model.table_name, ls=linestyles[i], c=colors[i], alpha=0.5)
        axes[0].fill_between(confounding_strengths, mrr_mean - mrr_err, mrr_mean + mrr_err, alpha=0.1, color=colors[i])

        ipw_mrr_mean = np.mean(ipw_mrrs, axis=1)
        ipw_mrr_err = stats.sem(ipw_mrrs, axis=1)
        axes[0].plot(confounding_strengths, ipw_mrr_mean, label='IPW ' + model.table_name, ls=linestyles[i],
                     linewidth=3, c=colors[i])
        axes[0].fill_between(confounding_strengths, ipw_mrr_mean - ipw_mrr_err, ipw_mrr_mean + ipw_mrr_err, alpha=0.1,
                             color=colors[i])

        mcar_mrr_mean = np.mean(mcar_mrrs, axis=1)
        mcar_mrr_err = stats.sem(mcar_mrrs, axis=1)
        axes[1].plot(confounding_strengths, mcar_mrr_mean, label=model.table_name, ls=linestyles[i], c=colors[i],
                     alpha=0.5)
        axes[1].fill_between(confounding_strengths, mcar_mrr_mean - mcar_mrr_err, mcar_mrr_mean + mcar_mrr_err,
                             alpha=0.1, color=colors[i])

        ipw_mcar_mrr_mean = np.mean(ipw_mcar_mrrs, axis=1)
        ipw_mcar_mrr_err = stats.sem(ipw_mcar_mrrs, axis=1)
        axes[1].plot(confounding_strengths, ipw_mcar_mrr_mean, label='IPW ' + model.table_name, ls=linestyles[i],
                     linewidth=3, c=colors[i])
        axes[1].fill_between(confounding_strengths, ipw_mcar_mrr_mean - ipw_mcar_mrr_err,
                             ipw_mcar_mrr_mean + ipw_mcar_mrr_err, alpha=0.1, color=colors[i])

    plt.figure(fig.number)
    axes[0].set_title('Confounded')
    axes[1].set_title('Counterfactual')
    axes[0].set_ylabel('Prediction Quality')

    for i in range(2):
        axes[i].set_xlabel('Confounding Strength')

    axes[1].text(0, 0.87, 'MCDM', color=colors[0])
    axes[1].text(0, 0.805, 'CDM', color=colors[1])
    axes[1].text(0, 0.785, 'MNL', color=colors[2])
    axes[1].text(0, 0.72, 'Logit', color=colors[3])

    axes[1].text(8, 0.785, 'IPW', weight='bold', ha='right', color=colors[2])
    axes[1].text(8, 0.755, 'no IPW', alpha=0.7, ha='right', color=colors[2])

    axes[1].set_ylim(top=0.885)

    plt.subplots_adjust(wspace=0.05)

    plt.savefig(f'plots/synthetic_confounded_context_rand_{embedding_dim}_{context_strength}.pdf', bbox_inches='tight')
    plt.close()


def examine_lcl_ipw():
    item_feature_datasets = [Expedia]

    feat_names = {Expedia: ['Star Rating', 'Review Score', 'Location Score', 'Price', 'On Promotion']}

    short_feat_names = {Expedia: ['SR', 'RS', 'LS', '$', 'OP']}

    for col, dataset in enumerate(item_feature_datasets):
        fig, axes = plt.subplots(2, 2, figsize=(7, 4.5), sharex='all', sharey='all')

        results = torch.load(f'{RESULTS_DIR}/{dataset.name}_compare_models_full_data.pt')

        best_losses = dict()
        best_mrrs = dict()
        models = set()
        best_num_params = dict()
        best_state_dict = dict()

        for args, state_dict, loss, mrr, num_params in results:
            dataset, learning_rate, split_seed, model, ipw = args[:5]

            models.add(model)

            if (model, ipw) not in best_losses or loss < best_losses[model, ipw]:
                best_losses[model, ipw] = loss
                best_mrrs[model, ipw] = mrr
                best_num_params[model, ipw] = num_params
                best_state_dict[model, ipw] = state_dict

        all_contexts = dict()
        for i, model in enumerate([ConditionalLogit, LCL, ConditionalMultinomialLogit, MultinomialLCL]):
            row = i // 2
            col = i % 2

            for ipw in [False, True]:
                theta = best_state_dict[model, ipw]['theta'].numpy().squeeze()

                if model in [LCL, MultinomialLCL]:
                    contexts = best_state_dict[model, ipw]['A'].numpy()

                    all_contexts[model, ipw] = contexts

                xs = np.arange(len(feat_names[dataset])) + (-0.2 if ipw else 0.2)

                axes[row, col].barh(xs, theta, height=0.4, label='IPW' if ipw else 'no IPW', hatch='/////' if ipw else None, facecolor='blue' if ipw else 'red')

            axes[row, col].set_yticks(np.arange(len(feat_names[dataset])))
            axes[row, col].set_yticklabels(feat_names[dataset], ha='center')

            if row == 1:
                axes[row, col].set_xlabel('Coefficient')

            axes[row, col].vlines(0, -0.4, len(feat_names[dataset]) - 0.6, color='black', linewidth=1)
            axes[row, col].set_ylim(-0.4, len(feat_names[dataset]) - 0.6)

            axes[row, col].spines['right'].set_visible(False)
            axes[row, col].spines['top'].set_visible(False)
            axes[row, col].spines['left'].set_visible(False)
            axes[row, col].tick_params(axis='y', which='both', length=0, pad=40)
            axes[row, col].yaxis.tick_right()

            if row == 0:
                axes[row, col].set_title(model.table_name)

        axes[0, 0].legend(loc='lower left')

        axes[0, 1].text(0.7, 2, 'No Regression', rotation=270, ha='center', va='center', fontsize=12)
        axes[1, 1].text(0.7, 2, 'Regression', rotation=270, ha='center', va='center', fontsize=12)

        plt.subplots_adjust(wspace=0.5, hspace=0.1 if dataset == Expedia else 0.3)
        plt.savefig(f'plots/{dataset.name}-pref-coeffs.pdf', bbox_inches='tight')
        plt.close()

        fig, axes = plt.subplots(2, 2, figsize=(3, 3))

        all_context_vals = np.array(list(all_contexts.values())).flatten()
        max_abs_context = max(abs(min(all_context_vals)), abs(max(all_context_vals)))

        for i, model in enumerate([LCL, MultinomialLCL]):
            for j, ipw in enumerate([False, True]):
                im = axes[i, j].imshow(all_contexts[model, ipw], cmap='bwr', vmin=-max_abs_context,
                                       vmax=max_abs_context)

                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])

        if dataset == Expedia:
            for i in range(2):
                axes[i, 0].set_yticks(range(len(feat_names[dataset])))
                axes[i, 0].set_yticklabels(feat_names[dataset])

                axes[1, i].set_xticks(range(len(feat_names[dataset])))
                axes[1, i].set_xticklabels(short_feat_names[dataset])

        axes[0, 0].text(0.5, 1.05, 'No IPW', transform=axes[0, 0].transAxes, ha='center', fontsize=12)
        axes[0, 1].text(0.5, 1.05, 'IPW', transform=axes[0, 1].transAxes, ha='center', fontsize=12)

        axes[0, 1].text(1.08, 0.5, 'LCL', rotation=270, transform=axes[0, 1].transAxes, ha='center', va='center',
                        fontsize=12)
        axes[1, 1].text(1.08, 0.5, 'MLCL', rotation=270, transform=axes[1, 1].transAxes, ha='center', va='center',
                        fontsize=12)

        cax = plt.axes([1, 0.25, 0.05, 0.5])
        plt.colorbar(im, cax=cax)

        plt.subplots_adjust(wspace=0.05, hspace=0.05)

        plt.savefig(f'plots/{dataset.name}-context-matrix.pdf', bbox_inches='tight')
        plt.close()

        config = [
            (0, (1, 0), (1, 1), 'Covariates', '#5ebf72'),
            (0, (0, 0), (1, 0), 'Context Effects', '#a3469d'),
            (1, (0, 1), (1, 1), None, '#a3469d'),
            (1, (0, 0), (0, 1), None, '#5ebf72'),
        ]

        fig, axes = plt.subplots(1, 2, figsize=(5, 4), sharey='row')

        for model in [ConditionalLogit, ConditionalMultinomialLogit, LCL, MultinomialLCL]:
            print(f'{model.table_name} & ${-int(best_losses[model, False])}$ & ${-int(best_losses[model, True]/dataset.get_ipw_weights().mean().item())}$\\\\')

        for col, ipw in enumerate([False, True]):

            losses = np.array([[best_losses[ConditionalLogit, ipw], best_losses[ConditionalMultinomialLogit, ipw]],
                               [best_losses[LCL, ipw], best_losses[MultinomialLCL, ipw]]])

            if ipw:
                losses /= dataset.get_ipw_weights().mean().item()

            num_params = np.array(
                [[best_num_params[ConditionalLogit, ipw], best_num_params[ConditionalMultinomialLogit, ipw]],
                 [best_num_params[LCL, ipw], best_num_params[MultinomialLCL, ipw]]])

            gap = losses[0, 0] - losses[1, 1]


            axes[col].text(0.5, losses[0, 0] + gap * 0.015, 'Cond. Logit', ha='center', va='bottom')
            axes[col].text(0.5, losses[1, 1] - gap * 0.015, 'MLCL', ha='center', va='top')

            for x, top, bottom, label, color in config:
                axes[col].bar([x], losses[top] - losses[bottom], bottom=losses[bottom], label=label, color=color)
                p = chi2.sf(2 * (losses[top] - losses[bottom]), num_params[bottom] - num_params[top])

                p = f'{p:.2g}' if p > 10 ** -10 else '$< 10^{-10}$'
                axes[col].text(x, losses[top] - (losses[top] - losses[bottom]) / 2, p, ha='center', va='center',
                                      color='white')

            axes[col].hlines([losses[0, 0], losses[1, 1]], -0.5, 1.5, color='black')

            axes[col].hlines(losses[1, 0], -0.5, 0.5, color='black')
            axes[col].hlines(losses[0, 1], 0.5, 1.5, color='black')

            axes[col].text(.1, losses[1, 0] - gap * 0.015, 'LCL', ha='left', va='top')
            axes[col].text(0.8, losses[0, 1] + gap * 0.015, 'MCL', ha='right', va='bottom')

            axes[col].set_xticks([])
            axes[col].set_title('IPW' if ipw else 'No IPW')

        axes[-1].legend(bbox_to_anchor=(1, -0.05), loc='upper right')
        axes[0].set_ylabel('Negative Log-Likelihood')
        fig.tight_layout()
        plt.savefig(f'plots/{dataset.name}-ipw-bars.pdf', bbox_inches='tight')
        plt.close()


def plot_yoochoose_clustering():
    cluster_counts = range(2, 11)
    seeds = range(8)
    lrs = np.logspace(np.log10(0.001), np.log10(0.1), 3)

    colors = ["#8a5d8f",
              "#9ebf51",
              "#8d52c8",
              "#b56042",
              "#7e9182"]

    for dataset in [YoochooseCats]:
        with open(f'{RESULTS_DIR}/{dataset.name}_compare_models_full_data.pt', 'rb') as f:
            results = torch.load(f)

        best_losses = dict()
        best_mrrs = dict()
        best_num_params = dict()

        for args, state_dict, loss, mrr, num_params in results:
            dataset, learning_rate, split_seed, model, use_ipw = args[:5]

            if model not in best_losses or loss < best_losses[model]:
                best_losses[model] = loss
                best_mrrs[model] = mrr
                best_num_params[model] = num_params

        with open(f'{RESULTS_DIR}/{dataset.name}_mixed_logit_em.pt', 'rb') as f:
            em_results = torch.load(f)

        em_losses = [y[1] for y in sorted(em_results, key=lambda x: x[0][1]) if not y[0][2]]
        em_spec_init_losses = [y[1] for y in sorted(em_results, key=lambda x: x[0][1]) if y[0][2]]

        with open(f'{RESULTS_DIR}/{dataset.name}_spectral_clustering.pt', 'rb') as f:
            cluster_results = torch.load(f)

        cluster_dict = dict()
        for args, spec_results, rand_results in cluster_results:
            cluster_dict[args] = spec_results, rand_results

        spec_losses = []
        rand_losses = []

        for n_clusters in cluster_counts:
            spec_losses.append([])
            rand_losses.append([])

            for seed in seeds:
                min_total_spec_loss = np.inf
                min_total_rand_loss = np.inf

                for lr in lrs:
                    total_spec_loss = 0
                    total_rand_loss = 0

                    for cluster in range(n_clusters):
                        total_spec_loss += cluster_dict[dataset, lr, seed, Logit, n_clusters][0][cluster][1]
                        total_rand_loss += cluster_dict[dataset, lr, seed, Logit, n_clusters][1][cluster][1]

                    min_total_spec_loss = min(min_total_spec_loss, total_spec_loss)
                    min_total_rand_loss = min(min_total_rand_loss, total_rand_loss)

                spec_losses[-1].append(min_total_spec_loss)
                rand_losses[-1].append(min_total_rand_loss)

        spec_losses = -np.array(spec_losses)
        rand_losses = -np.array(rand_losses)

        plt.figure(figsize=(4, 3))
        plt.hlines(-best_losses[Logit], min(cluster_counts), max(cluster_counts), label='Logit', color=colors[0], ls='--')
        plt.text(10, -252450, 'Logit', color=colors[0], fontsize=12, ha='right')

        plt.plot(cluster_counts, -np.array(em_losses), label='Mixed logit', color=colors[2])
        plt.text(10, -251500, 'Mixed logit', color=colors[2], fontsize=12, ha='right')

        mean_spec = np.mean(spec_losses, 1)
        err_spec = np.std(spec_losses, 1)
        plt.plot(cluster_counts, mean_spec, label='Spectral clustering', color=colors[3])
        plt.text(10, -249180, 'Spectral cluster logit', color=colors[3], fontsize=12, ha='right')
        plt.ylim(bottom=-252600, top=-248800)
        plt.fill_between(cluster_counts, mean_spec - err_spec, mean_spec + err_spec, alpha=0.1, color=colors[3])

        mean_rand = np.mean(rand_losses, 1)
        err_rand = np.std(rand_losses, 1)
        plt.plot(cluster_counts, mean_rand, label='Random clustering', color=colors[1])
        plt.text(2, -252450, 'Random cluster logit', color=colors[1], fontsize=12, ha='left')

        plt.fill_between(cluster_counts, mean_rand - err_rand, mean_rand + err_rand, alpha=0.1, color=colors[1])

        plt.xlabel('Cluster Count')
        plt.ylabel('Log-Likelihood')

        # plt.show()
        plt.savefig(f'plots/{dataset.name}_spectral_clustering.pdf', bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    os.makedirs('plots/', exist_ok=True)

    print('\n\nTABLE 3\n----------------')
    test_regularity(SFWork)

    print('\n\nTABLE 4\n----------------')
    test_regularity(SFShop)

    print('\n\nTABLE 5\n----------------')
    compare_likelihoods([SFWork, SFShop])
    compare_likelihoods([Expedia])

    print('\n\nTABLE 6\n----------------')
    examine_lcl_ipw()

    plot_synthetic_pets_ipw()

    plot_yoochoose_clustering()

    plot_synthetic_confounded_cdm()

