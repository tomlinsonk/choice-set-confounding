import time
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from scipy.stats import stats
from tqdm import tqdm

from datasets import SFShop, YoochooseCats, SFWork

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


# From https://github.com/pytorch/pytorch/issues/31829
@torch.jit.script
def logsumexp(x: torch.Tensor, dim: int) -> torch.Tensor:
    m, _ = x.max(dim=dim)
    mask = m == -float('inf')

    s = (x - m.masked_fill_(mask, 0).unsqueeze(dim=dim)).exp().sum(dim=dim)
    return s.masked_fill_(mask, 1).log() + m.masked_fill_(mask, -float('inf'))


class ChoiceModel(nn.Module):
    def loss(self, y_hat, y, weights):
        """
        The error in inferred log-probabilities given observations
        :param y_hat: log(choice probabilities)
        :param y: observed choices
        :return: the loss
        """
        return torch.dot(nnf.nll_loss(y_hat, y, reduction='none').squeeze(), weights) / len(y)

    def accuracy(self, y_hat, y):
        """
        Compute accuracy (fraction of choice set correctly predicted)
        :param y_hat: log(choice probabilities)
        :param y: observed choices
        :return: the accuracy
        """
        return (y_hat.argmax(1).int() == y.int()).float().mean()

    def mean_relative_rank(self, y_hat, y):
        """
        Compute mean rank of correct answer in output sorted by probability
        :param y_hat:
        :param y:
        :return:
        """
        return np.mean(self.relative_ranks(y_hat, y))

    def relative_ranks(self, y_hat, y):
        """
        Compute mean rank of correct answer in output sorted by probability
        :param y_hat:
        :param y:
        :return:
        """
        y_hat = y_hat.squeeze().cpu()
        y = y.squeeze().cpu()

        choice_set_lengths = np.array((~torch.isinf(y_hat)).sum(1))
        ranks = stats.rankdata(-y_hat.detach().numpy(), method='average', axis=1)[np.arange(len(y)), y] - 1

        return ranks / (choice_set_lengths - 1)

    @property
    def num_params(self):
        return sum(param.numel() for param in self.parameters() if param.requires_grad)


class ItemIdentityChoiceModel(ChoiceModel, ABC):
    @abstractmethod
    def forward(self, choice_sets):
        """
        Compute log(choice probabilities) of items in choice sets
        :param choice_sets: the choice sets
        :return: log(choice probabilities) over every choice set
        """
        pass


class ChooserFeatureChoiceModel(ChoiceModel, ABC):
    @abstractmethod
    def forward(self, choice_sets, chooser_features):
        """
        Compute log(choice probabilities) of items in choice sets
        :param choice_sets: the choice sets
        :param chooser_features: features of the chooser in each sample
        :return: log(choice probabilities) over every choice set
        """
        pass


class ItemFeatureChoiceModel(ChoiceModel, ABC):
    @abstractmethod
    def forward(self, choice_set_features, choice_set_lengths):
        """
        Compute log(choice probabilities) of items in choice sets
        :param choice_set_features: the choice sets, with item features
        :param choice_set_lengths: number of items in each choice set
        :return: log(choice probabilities) over every choice set
        """


class ItemChooserFeatureChoiceModel(ChoiceModel, ABC):
    @abstractmethod
    def forward(self, choice_set_features, choice_set_lengths, chooser_features):
        """
        Compute log(choice probabilities) of items in choice sets
        :param choice_set_features: the choice sets, with item features
        :param choice_set_lengths: number of items in each choice set
        :param chooser_features: features of the chooser in each sample
        :return: log(choice probabilities) over every choice set
        """
        pass


class LowRankCDM(ItemIdentityChoiceModel):
    """
    Implementation of low-rank CDM.
    Adapted from https://github.com/arjunsesh/cdm-icml.
    """
    name = 'low-rank-cdm'

    def __init__(self, num_items, rank=2):
        """
        Initialize a low rank CDM model for inference
        :param num_items: size of U
        :param rank: the rank of the CDM
        """
        super().__init__()
        self.num_items = num_items
        self.embedding_dim = rank

        self.target_embedding = nn.Parameter(torch.randn((self.num_items, self.embedding_dim)))
        self.context_embedding = nn.Parameter(torch.randn((self.num_items, self.embedding_dim)))

    def forward(self, choice_sets):
        """
        Compute log(choice probabilities) of items in choice sets
        :param choice_sets: the choice sets
        :return: log(choice probabilities) over every choice set
        """
        context_vecs = self.context_embedding * choice_sets
        target_vecs = self.target_embedding * choice_sets

        context_sums = context_vecs.sum(-2, keepdim=True) - context_vecs
        utilities = (target_vecs * context_sums).sum(-1, keepdim=True)
        utilities[choice_sets == 0] = -np.inf

        return nnf.log_softmax(utilities, 1)


class Logit(ItemIdentityChoiceModel):
    name = 'logit'
    table_name = 'Logit'

    def __init__(self, num_items):
        """
        Initialize an MNL model for inference
        :param num_items: size of U
        """
        super().__init__()
        self.num_items = num_items

        self.utilities = nn.Parameter(torch.zeros(self.num_items, 1))

    def forward(self, choice_sets):
        """
        Compute log(choice probabilities) of items in choice sets
        :param choice_sets: the choice sets
        :return: log(choice probabilities) over every choice set
        """

        utilities = self.utilities * choice_sets
        utilities[choice_sets == 0] = -np.inf

        return nnf.log_softmax(utilities, 1)


class CDM(ItemIdentityChoiceModel):
    name = 'cdm'
    table_name = 'CDM'

    def __init__(self, num_items):
        super().__init__()
        self.num_items = num_items
        self.pulls = nn.Parameter(torch.zeros(self.num_items, self.num_items))

    def forward(self, choice_sets):
        """
        Compute log(choice probabilities) of items in choice sets
        :param choice_sets: the choice sets
        :return: log(choice probabilities) over every choice set
        """
        utilities = ((choice_sets.squeeze() @ self.pulls).unsqueeze(-1) - torch.diag(self.pulls)[:, None]) * choice_sets
        utilities[choice_sets == 0] = -np.inf

        return nnf.log_softmax(utilities, 1)


class SingleContextCDM(ItemIdentityChoiceModel):
    name = 'single-context-cdm'

    def __init__(self, num_items, context_item):
        super().__init__()
        self.num_items = num_items
        self.context_item = context_item
        self.utilities = nn.Parameter(torch.zeros(self.num_items))
        self.pulls = nn.Parameter(torch.zeros(self.num_items))

    def forward(self, choice_sets):
        """
        Compute log(choice probabilities) of items in choice sets
        :param choice_sets: the choice sets
        :return: log(choice probabilities) over every choice set
        """
        utilities = (self.utilities + (self.pulls * choice_sets[:, self.context_item])).unsqueeze(-1)
        utilities[choice_sets == 0] = -np.inf

        return nnf.log_softmax(utilities, 1)


class MultinomialLogit(ChooserFeatureChoiceModel):
    name = 'multinomial-logit'
    table_name = 'MNL'

    def __init__(self, num_items, num_chooser_features):
        """
        Initialize an MNL model for inference
        :param num_items: size of U
        :param num_chooser_features: number of chooser features
        """
        super().__init__()
        self.num_chooser_features = num_chooser_features
        self.num_items = num_items
        self.coeffs = nn.Parameter(torch.ones(self.num_items, num_chooser_features))
        self.intercepts = nn.Parameter(torch.zeros(self.num_items, 1))

    def forward(self, choice_sets, chooser_features):
        """
        Compute log(choice probabilities) of items in choice sets
        :param choice_sets: the choice sets
        :param chooser_features: features of the chooser in each sample
        :return: log(choice probabilities) over every choice set
        """

        utilities = (self.coeffs * chooser_features[:, None, :]).sum(axis=2, keepdim=True) + self.intercepts
        utilities = utilities * choice_sets
        utilities[choice_sets == 0] = -np.inf

        return nnf.log_softmax(utilities, 1)


class MultinomialCDM(ChooserFeatureChoiceModel):
    name = 'multinomial-cdm'
    table_name = 'MCDM'

    def __init__(self, num_items, num_chooser_features):
        super().__init__()
        self.num_chooser_features = num_chooser_features
        self.num_items = num_items
        self.coeffs = nn.Parameter(torch.ones(self.num_items, num_chooser_features))
        self.pulls = nn.Parameter(torch.zeros(self.num_items, self.num_items))

    def forward(self, choice_sets, chooser_features):
        """
        Compute log(choice probabilities) of items in choice sets
        :param choice_sets: the choice sets
        :param chooser_features: features of the chooser in each sample
        :return: log(choice probabilities) over every choice set
        """

        utilities = (self.coeffs * chooser_features[:, None, :]).sum(axis=2, keepdim=True)
        utilities = utilities + (choice_sets.squeeze() @ self.pulls).unsqueeze(-1) - torch.diag(self.pulls)[:, None]
        utilities = utilities * choice_sets
        utilities[choice_sets == 0] = -np.inf

        return nnf.log_softmax(utilities, 1)


class MultinomialLowRankCDM(ChooserFeatureChoiceModel):
    name = 'multinomial-low-rank-cdm'
    table_name = 'rank-2 MCDM'

    def __init__(self, num_items, num_chooser_features, rank=2):
        super().__init__()
        self.num_chooser_features = num_chooser_features
        self.num_items = num_items
        self.coeffs = nn.Parameter(torch.ones(self.num_items, num_chooser_features))
        self.embedding_dim = rank

        self.target_embedding = nn.Parameter(torch.randn((self.num_items, self.embedding_dim)))
        self.context_embedding = nn.Parameter(torch.randn((self.num_items, self.embedding_dim)))

    def forward(self, choice_sets, chooser_features):
        """
        Compute log(choice probabilities) of items in choice sets
        :param choice_sets: the choice sets
        :param chooser_features: features of the chooser in each sample
        :return: log(choice probabilities) over every choice set
        """
        context_vecs = self.context_embedding * choice_sets
        target_vecs = self.target_embedding * choice_sets

        context_sums = context_vecs.sum(-2, keepdim=True) - context_vecs

        utilities = (self.coeffs * chooser_features[:, None, :]).sum(axis=2, keepdim=True)
        utilities = utilities * choice_sets

        utilities = utilities + (target_vecs * context_sums).sum(-1, keepdim=True)

        utilities[choice_sets == 0] = -np.inf

        return nnf.log_softmax(utilities, 1)


class ConditionalLogit(ItemFeatureChoiceModel):
    name = 'conditional-logit'
    table_name = 'CL'

    def __init__(self, num_item_feats):
        """
        :param num_item_feats: number of item features
        """
        super().__init__()
        self.num_item_feats = num_item_feats
        self.theta = nn.Parameter(torch.zeros(self.num_item_feats))

    def forward(self, choice_set_features, choice_set_lengths):
        """
        Compute log(choice probabilities) of items in choice sets
        :param choice_set_features: the choice sets, with item features
        :param choice_set_lengths: number of items in each cchoice set
        :return: log(choice probabilities) over every choice set
        """
        batch_size, max_choice_set_len, num_feats = choice_set_features.size()

        utilities = (self.theta * choice_set_features).sum(-1)
        utilities[torch.arange(max_choice_set_len)[None, :].to(device) >= choice_set_lengths[:, None]] = -np.inf

        return nnf.log_softmax(utilities, 1)


class ConditionalMultinomialLogit(ItemChooserFeatureChoiceModel):
    name = 'conditional-multinomial-logit'
    table_name = 'CML'

    def __init__(self, num_item_feats, num_chooser_feats):
        """
        :param num_item_feats: number of item features
        """
        super().__init__()
        self.num_item_feats = num_item_feats
        self.num_chooser_feats = num_chooser_feats
        self.theta = nn.Parameter(torch.zeros(self.num_item_feats, 1))
        self.B = nn.Parameter(torch.zeros(self.num_item_feats, self.num_chooser_feats))

    def forward(self, choice_set_features, choice_set_lengths, chooser_features):
        """
        Compute log(choice probabilities) of items in choice sets
        :param choice_set_features: the choice sets, with item features
        :param choice_set_lengths: number of items in each choice set
        :param chooser_features: features of the chooser in each sample
        :return: log(choice probabilities) over every choice set
        """
        batch_size, max_choice_set_len, num_feats = choice_set_features.size()

        utilities = ((self.theta + self.B @ chooser_features.unsqueeze(-1))[:, None, :, 0] * choice_set_features).sum(-1)
        utilities[torch.arange(max_choice_set_len)[None, :].to(device) >= choice_set_lengths[:, None]] = -np.inf

        return nnf.log_softmax(utilities, 1)


class MLPConditionalMultinomialLogit(ItemChooserFeatureChoiceModel):
    name = 'mlp-conditional-multinomial-logit'
    table_name = 'MLP-CML'

    def __init__(self, num_item_feats, num_chooser_feats):
        """
        :param num_item_feats: number of item features
        """
        super().__init__()
        self.num_item_feats = num_item_feats
        self.num_chooser_feats = num_chooser_feats
        self.theta = nn.Parameter(torch.zeros(self.num_item_feats, 1))
        self.mlp = nn.Sequential(
            nn.Linear(self.num_chooser_feats, self.num_chooser_feats),
            nn.ReLU(),
            nn.Linear(self.num_chooser_feats, self.num_item_feats),
            nn.ReLU(),
            nn.Linear(self.num_item_feats, self.num_item_feats)
        )

    def forward(self, choice_set_features, choice_set_lengths, chooser_features):
        """
        Compute log(choice probabilities) of items in choice sets
        :param choice_set_features: the choice sets, with item features
        :param choice_set_lengths: number of items in each choice set
        :param chooser_features: features of the chooser in each sample
        :return: log(choice probabilities) over every choice set
        """
        batch_size, max_choice_set_len, num_feats = choice_set_features.size()

        utilities = ((self.theta + self.mlp(chooser_features).unsqueeze(-1))[:, None, :, 0] * choice_set_features).sum(-1)
        utilities[torch.arange(max_choice_set_len)[None, :] >= choice_set_lengths[:, None]] = -np.inf

        return nnf.log_softmax(utilities, 1)


class LCL(ItemFeatureChoiceModel):
    name = 'lcl'
    table_name = 'LCL'

    def __init__(self, num_item_feats):
        """
        :param num_item_feats: number of item features
        """
        super().__init__()
        self.num_item_feats = num_item_feats
        self.theta = nn.Parameter(torch.zeros(self.num_item_feats))
        self.A = nn.Parameter(torch.zeros(self.num_item_feats, self.num_item_feats))

    def forward(self, choice_set_features, choice_set_lengths):
        batch_size, max_choice_set_len, num_feats = choice_set_features.size()

        # Compute mean of each feature over each choice set
        mean_choice_set_features = choice_set_features.sum(1) / choice_set_lengths.unsqueeze(-1)

        # Compute context effect in each sample
        context_effects = self.A @ mean_choice_set_features.unsqueeze(-1)

        # Compute context-adjusted utility of every item
        utilities = ((self.theta.unsqueeze(-1) + context_effects).view(batch_size, 1, -1) * choice_set_features).sum(-1)
        utilities[torch.arange(max_choice_set_len).to(device).unsqueeze(0) >= choice_set_lengths.unsqueeze(-1)] = -np.inf

        return nn.functional.log_softmax(utilities, 1)


class MultinomialLCL(ItemChooserFeatureChoiceModel):
    name = 'multinomial-lcl'
    table_name = 'MLCL'

    def __init__(self, num_item_feats, num_chooser_feats):
        """
        :param num_item_feats: number of item features
        """
        super().__init__()
        self.num_item_feats = num_item_feats
        self.num_chooser_feats = num_chooser_feats
        self.theta = nn.Parameter(torch.zeros(self.num_item_feats))
        self.A = nn.Parameter(torch.zeros(self.num_item_feats, self.num_item_feats))
        self.B = nn.Parameter(torch.zeros(self.num_item_feats, self.num_chooser_feats))

    def forward(self, choice_set_features, choice_set_lengths, chooser_features):
        batch_size, max_choice_set_len, num_feats = choice_set_features.size()

        # Compute mean of each feature over each choice set
        mean_choice_set_features = choice_set_features.sum(1) / choice_set_lengths.unsqueeze(-1)

        # Compute context effect in each sample
        context_effects = self.A @ mean_choice_set_features.unsqueeze(-1)

        # Compute per-chooser effect on utilities
        chooser_prefs = self.B @ chooser_features.unsqueeze(-1)

        # Compute context-adjusted utility of every item
        utilities = ((self.theta.unsqueeze(-1) + context_effects + chooser_prefs).view(batch_size, 1, -1) * choice_set_features).sum(-1)
        utilities[torch.arange(max_choice_set_len).to(device).unsqueeze(0) >= choice_set_lengths.unsqueeze(-1)] = -np.inf

        return nn.functional.log_softmax(utilities, 1)


class MLPMultinomialLCL(ItemChooserFeatureChoiceModel):
    name = 'mlp-multinomial-lcl'
    table_name = 'MLP-MLCL'

    def __init__(self, num_item_feats, num_chooser_feats):
        """
        :param num_item_feats: number of item features
        """
        super().__init__()
        self.num_item_feats = num_item_feats
        self.num_chooser_feats = num_chooser_feats
        self.theta = nn.Parameter(torch.zeros(self.num_item_feats))
        self.A = nn.Parameter(torch.zeros(self.num_item_feats, self.num_item_feats))
        self.mlp = nn.Sequential(
            nn.Linear(self.num_chooser_feats, self.num_chooser_feats),
            nn.ReLU(),
            nn.Linear(self.num_chooser_feats, self.num_item_feats),
            nn.ReLU(),
            nn.Linear(self.num_item_feats, self.num_item_feats)
        )

    def forward(self, choice_set_features, choice_set_lengths, chooser_features):
        batch_size, max_choice_set_len, num_feats = choice_set_features.size()

        # Compute mean of each feature over each choice set
        mean_choice_set_features = choice_set_features.sum(1) / choice_set_lengths.unsqueeze(-1)

        # Compute context effect in each sample
        context_effects = self.A @ mean_choice_set_features.unsqueeze(-1)

        # Compute per-chooser effect on utilities
        chooser_prefs = self.mlp(chooser_features).unsqueeze(-1)

        # Compute context-adjusted utility of every item
        utilities = ((self.theta.unsqueeze(-1) + context_effects + chooser_prefs).view(batch_size, 1, -1) * choice_set_features).sum(-1)
        utilities[torch.arange(max_choice_set_len).to(device).unsqueeze(0) >= choice_set_lengths.unsqueeze(-1)] = -np.inf

        return nn.functional.log_softmax(utilities, 1)


class MixedLogit(ItemIdentityChoiceModel):

    name = 'mixed-logit'

    def __init__(self, num_items, num_components):
        super().__init__()

        self.num_items = num_items
        self.num_components = num_components

        self.utilities = nn.Parameter(torch.rand(self.num_items, self.num_components), requires_grad=True)
        self.mixture_weights = nn.Parameter(torch.ones(self.num_components), requires_grad=True)

    def forward(self, choice_sets):
        """
        Compute log(choice probabilities) of items in choice sets
        :param choice_sets: the choice sets
        :return: log(choice probabilities) over every choice set
        """

        utilities = self.utilities * choice_sets
        utilities[choice_sets.squeeze() == 0] = -np.inf

        log_probs = nnf.log_softmax(utilities, 1)

        # Combine the MNLs into single probability using weights
        # using the torch.logsumexp fix (https://github.com/pytorch/pytorch/issues/31829)
        return logsumexp(log_probs + torch.log_softmax(self.mixture_weights, 0), 2).unsqueeze(-1)


class EMAlgorithmQ(nn.Module):

    def __init__(self, responsibilities, utilities_init):
        super().__init__()

        self.responsibilities = responsibilities
        self.utilities = nn.Parameter(utilities_init, requires_grad=True)

    def forward(self, choice_sets, choices):
        samples = choices.size(0)

        # Compute utility of each item under each logit
        utilities = self.utilities * choice_sets
        utilities[choice_sets.squeeze() == 0] = -np.inf

        log_probs = nnf.log_softmax(utilities, 1)[torch.arange(samples), choices.squeeze()]

        return - (self.responsibilities * log_probs).sum()


def fit(model, data, epochs=500, learning_rate=5e-2, l2_lambda=1e-4, show_live_loss=False, show_progress=True):
    """
    Fit a choice model to data using the given optimizer.

    :param model: a nn.Module
    :param data:
    :param epochs: number of optimization epochs
    :param learning_rate: step size hyperparameter for Rprop
    :param l2_lambda: regularization hyperparameter
    :param show_live_loss: if True, add loss/accuracy to progressbar. Adds ~50% overhead
    """
    torch.set_num_threads(1)

    optimizer = torch.optim.Rprop(model.parameters(), lr=learning_rate)
    # torch.autograd.set_detect_anomaly(True)

    choices, weights = data[-2:]

    progress_bar = tqdm(range(epochs), total=epochs) if show_progress else range(epochs)

    for epoch in progress_bar:
        model.train()
        accuracies = []
        losses = []

        optimizer.zero_grad()

        loss = model.loss(model(*data[:-2]), choices, weights)

        l2_reg = torch.tensor(0.).to(device)
        for param in model.parameters():
            l2_reg += torch.pow(param, 2).sum()
        loss += l2_lambda * l2_reg

        loss.backward(retain_graph=None if epoch != epochs - 1 else True)

        with torch.no_grad():
            gradient = torch.stack([(item.grad ** 2).sum() for item in model.parameters()]).sum()

            if gradient.item() < 10 ** -8:
                break

        optimizer.step()

        if show_progress and show_live_loss:
            model.eval()
            with torch.no_grad():
                accuracy = model.accuracy(model(*data[:-2]), choices)
            losses.append(loss.detach())
            accuracies.append(accuracy)

        if show_progress and show_live_loss:
            progress_bar.set_description(f'Loss: {np.mean(losses):.4f}, Accuracy: {np.mean(accuracies):.4f}, Grad: {gradient.item():.3e}. Epochs')

    loss = model.loss(model(*data[:-2]), choices, weights)
    loss.backward()
    with torch.no_grad():
        gradient = torch.stack([(item.grad ** 2).sum() for item in model.parameters()]).sum()

    if show_progress:
        print('Done. Final gradient:', gradient.item(), 'Final NLL:', loss.item() * len(data[-1]))

    return loss.item() * len(data[-1])


def mixed_logit_em(model, choice_sets, choices, timeout_min=60):
    samples = choices.size(0)
    start_time = time.time()

    # Loop until timeout or gradient norm < 10^-6
    while True:
        utilities = model.utilities * choice_sets
        utilities[choice_sets.squeeze() == 0] = -np.inf

        log_probs = nnf.log_softmax(utilities, 1)[torch.arange(samples), choices.squeeze()]
        responsibilities = nnf.softmax(log_probs + torch.log_softmax(model.mixture_weights, 0), 1)

        Q = EMAlgorithmQ(responsibilities, model.utilities.data)

        optimizer = torch.optim.Rprop(Q.parameters(), lr=0.01)

        for epoch in range(100):
            loss = Q(choice_sets, choices)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)

            with torch.no_grad():
                gradient = torch.stack([(item.grad ** 2).sum() for item in Q.parameters()]).sum()

                if gradient.item() < 10 ** -7:
                    break

            if timeout_min is not None:
                if time.time() - start_time > timeout_min * 60:
                    break

            optimizer.step()

        if timeout_min is not None:
            if time.time() - start_time > timeout_min * 60:
                break

        model.mixture_weights.data = torch.log(responsibilities.mean(0))
        model.utilities.data = Q.utilities.data

        loss = nnf.nll_loss(model(choice_sets), choices, reduction='sum')

        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            gradient = torch.stack([(item.grad ** 2).sum() for item in model.parameters()]).sum()
            if gradient.item() < 10 ** -6:
                break

    return model, nnf.nll_loss(model(choice_sets), choices, reduction='sum').item()


if __name__ == '__main__':
    choice_sets, choices, person_df = SFWork.load_pytorch()
    model = Logit(len(choice_sets[0]))

    fit(model, (choice_sets, choices, torch.ones(len(choices))), l2_lambda=10**-4, show_live_loss=True)


