from abc import ABC, abstractmethod

import datasets
from scipy import stats
from sklearn import tree, preprocessing
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LogisticRegression



class ItemIdentityChoiceSetModel(ABC):
    dataset = None

    @abstractmethod
    def train(self, choice_sets, person_features):
        ...

    @abstractmethod
    def predict(self, person_features):
        ...

    @abstractmethod
    def choice_set_assignment_probs(self, person_features, choice_sets):
        ...

    def __repr__(self):
        return f'{self.__class__.__name__}({self.dataset})'


class ItemFeatureChoiceSetModel(ABC):
    dataset = None

    @abstractmethod
    def train(self, choice_set_features, choice_set_lengths, person_features):
        ...

    @abstractmethod
    def predict(self, person_features):
        ...

    @abstractmethod
    def choice_set_assignment_probs(self, person_features, choice_set_features, choice_set_lengths):
        ...

    def __repr__(self):
        return f'{self.__class__.__name__}({self.dataset})'


class ItemIdentityLogisticRegression(ItemIdentityChoiceSetModel):
    log_regs = []
    const_pred = []
    num_items = 0
    scaler = None

    def __init__(self, dataset):
        self.dataset = dataset
        self.num_items = len(dataset.item_names)
        self.log_regs = [LogisticRegression(random_state=0, max_iter=200, n_jobs=1) for _ in range(self.num_items)]
        self.const_pred = [None for _ in range(self.num_items)]

    def train(self, person_features, choice_sets):
        self.scaler = preprocessing.StandardScaler().fit(person_features)

        for i in range(self.num_items):
            if len(np.unique(choice_sets[:, i])) == 1:
                self.const_pred[i] = choice_sets[0, i]
                continue
            self.log_regs[i].fit(self.scaler.transform(person_features), choice_sets[:, i])

    def predict(self, person_features):
        preds = []
        for i in range(self.num_items):
            if self.const_pred[i] is not None:
                preds.append(np.full(len(person_features), self.const_pred[i]))
                continue
            preds.append(self.log_regs[i].predict(self.scaler.transform(person_features)))
        return np.column_stack(preds)

    def plot(self, feature_names):
        weights = np.array([model.intercept_ for model in self.log_regs])
        print(np.exp(weights))

    def log_choice_set_assignment_probs(self, person_features, choice_sets):
        n_samples = choice_sets.shape[0]
        probs = []

        for i in range(self.num_items):
            if self.const_pred[i] is not None:
                probs.append(np.full(n_samples, (self.const_pred[i] + 1) / 2))
                continue
            probs.append(self.log_regs[i].predict_proba(self.scaler.transform(person_features))[:, 1])

        probs = np.column_stack(probs)

        probs = np.abs((1 - choice_sets) / 2 - probs)
        log_probs = np.log(probs.clip(10**-16, 1)).sum(1)
        return log_probs

    def choice_set_assignment_probs(self, person_features, choice_sets):
        log_probs = self.log_choice_set_assignment_probs(person_features, choice_sets)
        return np.exp(log_probs)

    def item_assignment_probs(self, person_features, is_present, item):
        return np.abs(is_present - self.log_regs[item].predict_proba(self.scaler.transform(person_features))[:, 1])


class AffineGaussian(ItemFeatureChoiceSetModel):
    W = None
    beta = None
    cov = None

    def train(self, choice_set_features, choice_set_lengths, x_a):

        y_C = (choice_set_features.sum(1) / choice_set_lengths.unsqueeze(-1)).numpy()
        y_D = np.mean(y_C, 0)
        x_D = np.mean(x_a, 0)

        # sum of outer products of corresponding rows
        M1 = np.einsum('ij,ik->jk', y_C - y_D, x_a)
        M2 = np.linalg.inv(np.einsum('ij,ik->jk', x_a - x_D, x_a))

        self.W = M1 @ M2
        self.beta = y_D - self.W @ x_D

        err = y_C - np.einsum('ij,kj->ki', self.W, x_a) - self.beta
        self.cov = np.einsum('ij,ik->jk', err, err) / err.shape[0]

    def predict(self, x_a):
        return np.einsum('ij,kj->ki', self.W, x_a) + self.beta

    def log_choice_set_assignment_probs(self, x_a, choice_set_features, choice_set_lengths):
        y_C = (choice_set_features.sum(1) / choice_set_lengths.unsqueeze(-1)).numpy()

        probs = [stats.multivariate_normal.pdf(y_C[i], mean=self.W @ x_a[i] + self.beta, cov=self.cov, allow_singular=True) for i in
                 range(len(x_a))]

        return np.log(np.array(probs).clip(10**-16, 1))

    def choice_set_assignment_probs(self, x_a, choice_set_features, choice_set_lengths):
        return np.exp(self.log_choice_set_assignment_probs(x_a, choice_set_features, choice_set_lengths))


def get_item_probs(Model, dataset, item):
    choice_sets, choices, person_df = dataset.load()

    model = Model(dataset)
    model.train(person_df.values, (choice_sets * 2) - 1)

    return model.item_assignment_probs(person_df.values, choice_sets[:, item], item)
