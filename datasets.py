import os
import pickle
from abc import ABC, abstractmethod
from itertools import chain, combinations

import numpy as np
import pandas as pd
import scipy.io as sio
import torch
import yaml
from scipy.special import softmax
from sklearn import preprocessing
from tqdm import tqdm

import choice_set_models

with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

DATA_DIR = config['datadir']
IPW_DIR = 'ipw-weights'


def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))


def all_choice_set_indices(n_items):
    indices = list(range(n_items))
    return chain.from_iterable(combinations(indices, size) for size in range(2, n_items+1))


class Dataset(ABC):
    item_names = []

    @classmethod
    @abstractmethod
    def load(cls):
        ...

    @classmethod
    def one_hot_encode(cls, df, col_name):
        """
        One-hot encode a categorical feature in a pandas dataframe in-place
        :param df: the pandas DataFrame
        :param col_name: the name of the column
        """
        df = pd.concat([df, pd.get_dummies(df[col_name], prefix=col_name, drop_first=True)], axis=1)
        df.drop([col_name], axis=1, inplace=True)
        return df


class ItemIdentityDataset(Dataset, ABC):
    @classmethod
    def load_pytorch(cls):
        choice_sets, choices, person_df = cls.load()
        choice_sets = torch.from_numpy(choice_sets)[:, :, None]
        choices = torch.tensor(choices)

        if len(person_df.index) > 0:
            person_df = pd.DataFrame(preprocessing.StandardScaler().fit_transform(person_df), columns=person_df.columns, index=person_df.index)

        return choice_sets, choices, person_df

    @classmethod
    def get_choice_set_probs(cls, use_cached=True, log_probs=False):
        fname = f'{IPW_DIR}/{cls.__name__}_choice_set_log_probs.pt'
        if os.path.isfile(fname) and use_cached:
            with open(fname, 'rb') as f:
                log_choice_set_probs = torch.load(f)
        else:
            choice_sets, choices, person_df = cls.load()
            choice_sets = (2 * choice_sets) - 1

            model = choice_set_models.ItemIdentityLogisticRegression(cls)
            model.train(person_df.values, choice_sets)

            log_choice_set_probs = model.log_choice_set_assignment_probs(person_df.values, choice_sets)

            with open(fname, 'wb') as f:
                torch.save(log_choice_set_probs, f)

        return log_choice_set_probs if log_probs else np.exp(log_choice_set_probs)

    @classmethod
    def get_ipw_weights(cls, eps=0.01):
        set_probs = cls.get_choice_set_probs()
        w = 1 / (set_probs / np.max(set_probs) + eps)

        return torch.from_numpy(w).float()

    @classmethod
    def get_log_prob_ipw_weights(cls, max_weight=100):
        w = -cls.get_choice_set_probs(log_probs=True)
        w = np.clip(w - min(w) + 1, None, max_weight)
        return torch.from_numpy(w).float()


class ItemFeatureDataset(Dataset, ABC):
    @classmethod
    def load_pytorch(cls):
        choice_set_features, choice_set_lengths, choices, person_df = cls.load()
        choice_set_features = torch.from_numpy(choice_set_features).float()
        choice_set_lengths = torch.from_numpy(choice_set_lengths).long()
        choices = torch.tensor(choices).long()

        all_feature_vecs = choice_set_features[torch.arange(choice_set_features.size(1))[None, :] < choice_set_lengths[:, None]]
        means = all_feature_vecs.mean(0)
        stds = all_feature_vecs.std(0)

        choice_set_features[torch.arange(choice_set_features.size(1))[None, :] < choice_set_lengths[:, None]] -= means
        choice_set_features[torch.arange(choice_set_features.size(1))[None, :] < choice_set_lengths[:, None]] /= stds

        person_df = pd.DataFrame(preprocessing.StandardScaler().fit_transform(person_df), columns=person_df.columns, index=person_df.index)

        return choice_set_features, choice_set_lengths, choices, person_df

    @classmethod
    def get_choice_set_probs(cls, use_cached=True, log_probs=False):
        fname = f'{IPW_DIR}/{cls.__name__}_choice_set_log_probs.pt'
        if os.path.isfile(fname) and use_cached:
            with open(fname, 'rb') as f:
                log_choice_set_probs = torch.load(f)
        else:
            choice_set_features, choice_set_lengths, choices, person_df = cls.load_pytorch()

            choice_set_model = choice_set_models.AffineGaussian()
            choice_set_model.train(choice_set_features, choice_set_lengths, person_df.values)

            log_choice_set_probs = choice_set_model.log_choice_set_assignment_probs(person_df.values,
                                                                                    choice_set_features,
                                                                                    choice_set_lengths)

            with open(fname, 'wb') as f:
                torch.save(log_choice_set_probs, f)

        return log_choice_set_probs if log_probs else np.exp(log_choice_set_probs)

    @classmethod
    def get_ipw_weights(cls, eps=0.01):
        set_probs = cls.get_choice_set_probs()
        w = 1 / (set_probs / np.max(set_probs) + eps)

        return torch.from_numpy(w).float()

    @classmethod
    def get_log_prob_ipw_weights(cls, max_weight=100):
        w = -cls.get_choice_set_probs(log_probs=True)
        w = np.clip(w - min(w) + 1, None, max_weight)
        return torch.from_numpy(w).float()


class SFWork(ItemIdentityDataset):
    name = 'sf-work'

    item_names = ['Drive Alone', 'Shared Ride 2', 'Shared Ride 3+', 'Transit', 'Bike', 'Walk']

    @classmethod
    def load(cls):
        user_feature_names = ['femdum', 'age', 'corredis', 'dist', 'drlicdum', 'famtype', 'hhinc', 'hhowndum', 'hhsize',
                              'nm12to16', 'nm5to11', 'nmlt5', 'noncadum', 'numadlt', 'numemphh', 'numveh', 'rsempden',
                              'rspopden', 'vehavdum', 'vehbywrk', 'wkccbd', 'wkempden', 'wknccbd', 'wkpopden']

        sf_work = sio.loadmat(f'{DATA_DIR}/SF/SF-raw/SF_HBW/SFMTCWork6.mat')

        indivs = np.unique(np.concatenate((sf_work['hhid'], sf_work['perid']), axis=1), axis=0)
        indiv_indices = {(hhid, perid): [] for hhid, perid in indivs}

        for i in range(len(sf_work['hhid'])):
            indiv_indices[sf_work['hhid'][i][0], sf_work['perid'][i][0]].append(i)

        choice_sets = np.zeros((len(indivs), 6), dtype=int)
        for i, (hhid, perid) in enumerate(indivs):
            for j in indiv_indices[hhid, perid]:
                choice_sets[i, sf_work['alt'][j] - 1] = 1

        choices = np.zeros((len(indivs), 1), dtype=int)
        for i, (hhid, perid) in enumerate(indivs):
            for j in indiv_indices[hhid, perid]:
                if sf_work['chosen'][j]:
                    choices[i] = sf_work['alt'][j] - 1

        person_features = np.zeros((len(choice_sets), len(user_feature_names)), dtype=float)
        for i, (hhid, perid) in enumerate(indivs):
            j = indiv_indices[hhid, perid][0]
            for feat_idx, feat in enumerate(user_feature_names):
                person_features[i, feat_idx] = sf_work[feat][j]

        multi_idx = pd.MultiIndex.from_tuples([tuple(row) for row in indivs], names=('hhid', 'perid'))

        person_df = cls.one_hot_encode(pd.DataFrame(person_features, index=multi_idx, columns=user_feature_names), 'famtype')

        return choice_sets, choices, person_df


class SFShop(ItemIdentityDataset):
    name = 'sf-shop'

    item_names = ['Transit', 'SR2', 'SR3+',
                  'Drive Alone and SR', 'SR2 and SR3+', 'Bike', 'Walk', 'Drive Alone']

    @classmethod
    def load(cls):
        user_feature_names = ['DISTANCE', 'D_DENS', 'HHSIZE', 'HHSIZE5', 'INCOME', 'O_DENS', 'URBAN', 'VEHICLES']

        sf_shop = sio.loadmat(f'{DATA_DIR}/SF/SF-raw/SF_HBShO/SFHBSHOw5.mat')

        indivs = np.unique(sf_shop['ID'])
        indiv_indices = {id: [] for id in indivs}

        for i in range(len(sf_shop['ID'])):
            indiv_indices[sf_shop['ID'][i][0]].append(i)

        choice_sets = np.zeros((len(indivs), len(cls.item_names)), dtype=int)
        for i, id in enumerate(indivs):
            for j in indiv_indices[id]:
                choice_sets[i, sf_shop['ALTID'][j] - 1] = 1

        choices = np.zeros((len(indivs), 1), dtype=int)
        for i, id in enumerate(indivs):
            for j in indiv_indices[id]:
                if sf_shop['CHOSEN'][j]:
                    choices[i] = sf_shop['ALTID'][j] - 1

        person_features = np.zeros((len(choice_sets), len(user_feature_names)), dtype=float)
        for i, id in enumerate(indivs):
            j = indiv_indices[id][0]
            for feat_idx, feat in enumerate(user_feature_names):
                person_features[i, feat_idx] = sf_shop[feat][j]

        person_df = pd.DataFrame(person_features, index=indivs, columns=user_feature_names)

        return choice_sets, choices, person_df


class Sushi(ItemIdentityDataset):
    name = 'sushi'
    item_names = ['ebi', 'anago', 'maguro', 'ika', 'uni', 'tako', 'ikura', 'tamago', 'toro', 'amaebi', 'hotategai', 'tai',
             'akagai', 'hamachi', 'awabi', 'samon', 'kazunoko', 'shako', 'saba', 'chu_toro', 'hirame', 'aji', 'kani',
             'kohada', 'torigai', 'unagi', 'tekka_maki', 'kanpachi', 'mirugai', 'kappa_maki', 'geso', 'katsuo',
             'iwashi', 'hokkigai', 'shimaaji', 'kanimiso', 'engawa', 'negi_toro', 'nattou_maki', 'sayori',
             'takuwan_maki', 'botanebi', 'tobiko', 'inari', 'mentaiko', 'sarada', 'suzuki', 'tarabagani',
             'ume_shiso_maki', 'komochi_konbu', 'tarako', 'sazae', 'aoyagi', 'toro_samon', 'sanma', 'hamo', 'nasu',
             'shirauo', 'nattou', 'ankimo', 'kanpyo_maki', 'negi_toro_maki', 'gyusashi', 'hamaguri', 'basashi', 'fugu',
             'tsubugai', 'ana_kyu_maki', 'hiragai', 'okura', 'ume_maki', 'sarada_maki', 'mentaiko_maki', 'buri',
             'shiso_maki', 'ika_nattou', 'zuke', 'himo', 'kaiware', 'kurumaebi', 'mekabu', 'kue', 'sawara', 'sasami',
             'kujira', 'kamo', 'himo_kyu_maki', 'tobiuo', 'ishigakidai', 'mamakari', 'hoya', 'battera', 'kyabia',
             'karasumi', 'uni_kurage', 'karei', 'hiramasa', 'namako', 'shishamo', 'kaki']

    @classmethod
    def load(cls):
        user_feature_names = ['gender', 'age', 'survey_time', 'child_prefecture', 'child_region',
                              'child_east/west', 'prefecture', 'region', 'east/west', 'same_prefecture']

        rankings = np.loadtxt(f'{DATA_DIR}/sushi3-2016/sushi3b.5000.10.order', skiprows=1, usecols=range(2, 12),
                              dtype=int)

        person_features = np.loadtxt(f'{DATA_DIR}/sushi3-2016/sushi3.udata')

        person_df = pd.DataFrame(person_features[:, 1:], index=person_features[:, 0], columns=user_feature_names)

        categorical_feats = ['age', 'child_prefecture', 'child_region', 'prefecture', 'region']
        for feat in categorical_feats:
            person_df = cls.one_hot_encode(person_df, feat)

        choice_sets = np.zeros((len(rankings), 100), dtype=int)
        choice_sets[np.arange(len(rankings))[:, None], rankings] = 1
        choices = rankings[:, 0][:, None]

        return choice_sets, choices, person_df


class FeatureSushi(ItemFeatureDataset):
    name = 'feature-sushi'

    @classmethod
    def load(cls):
        old_choice_sets, old_choices, person_df = Sushi.load()

        item_feats = np.loadtxt(f'{DATA_DIR}/sushi3-2016/sushi3.idata', usecols=[2, 3, 5, 6, 7, 8])
        item_df = pd.DataFrame(item_feats, columns=['style', 'major_group', 'oiliness', 'popularity', 'price', 'availability'])

        range_100 = np.arange(100)
        choice_set_indices = np.array([range_100[row == 1] for row in old_choice_sets])
        choice_set_features = np.array([item_feats[row] for row in choice_set_indices])

        choice_set_lengths = np.full(len(choice_set_features), 10)

        choices = np.array([np.searchsorted(choice_set_indices[i], old_choices[i])[0] for i in range(len(choice_set_features))])

        return choice_set_features, choice_set_lengths, choices, person_df


class Expedia(ItemFeatureDataset):
    name = 'expedia'

    @classmethod
    def load(cls):
        pickle_fname = f'{DATA_DIR}/pickles/{cls.name}.pickle'
        if os.path.isfile(pickle_fname):
            with open(pickle_fname, 'rb') as f:
                return pickle.load(f)

        item_feats = ['prop_starrating', 'prop_review_score', 'prop_location_score1', 'price_usd', 'promotion_flag']
        chooser_feat_names = ['visitor_hist_starrating', 'visitor_hist_adr_usd', 'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool']

        df = pd.read_csv(f'{DATA_DIR}/expedia-personalized-sort/train.csv', usecols=['srch_id', 'prop_id', 'booking_bool', 'srch_destination_id'] + item_feats + chooser_feat_names)

        # Select only searches that result in a booking
        df = df[df.groupby(['srch_id'])['booking_bool'].transform(max) == 1]

        max_choice_set_size = df['srch_id'].value_counts().max()
        samples = df['srch_id'].nunique()
        n_feats = len(item_feats)

        choice_set_features = np.zeros((samples, max_choice_set_size, n_feats))
        choice_set_lengths = np.zeros(samples)
        choices = np.zeros(samples)

        chooser_feats = np.zeros((samples, len(chooser_feat_names)+1))

        for i, (srch_id, group) in tqdm(enumerate(df.groupby('srch_id')), total=samples):
            choice_set_length = len(group.index)
            choice_set_lengths[i] = choice_set_length

            item_features = group[item_feats].values
            item_features[np.isnan(item_features)] = 0

            choice_set_features[i, :choice_set_length] = item_features

            choices[i] = np.where(group['booking_bool'] == 1)[0]
            chooser_feats[i, :-1] = group[chooser_feat_names].values[0]

            # add 'has_prev_purchase' feature
            chooser_feats[i, -1] = int(not np.isnan(chooser_feats[i, 0]))

        # replace nan features with mean of non-nan
        for col in range(len(chooser_feat_names)):
            nan_idx = np.isnan(chooser_feats[:, col])
            mean = np.mean(chooser_feats[~nan_idx, col])
            chooser_feats[nan_idx, col] = mean

        person_df = pd.DataFrame(chooser_feats, columns=chooser_feat_names + ['has_prev_purchase'])

        with open(pickle_fname, 'wb') as f:
            pickle.dump((choice_set_features, choice_set_lengths, choices, person_df), f)

        return choice_set_features, choice_set_lengths, choices, person_df


class SyntheticPetsSuper(ItemIdentityDataset):
    item_names = ['cat', 'dog', 'bird']
    mnar = None
    choice_set_ignorability = None
    preference_ignorability = None

    @classmethod
    def generate(cls, samples, seed):
        np.random.seed(seed)

        n_items = len(cls.item_names)

        choice_sets = np.full((samples, n_items), 1)
        choices = np.zeros((samples, 1))

        person_feats = np.zeros((samples, 2))
        person_feats[:, 0] = (np.random.uniform(size=samples) < 0.25) * 1
        person_feats[:, 1] = person_feats[:, 0]
        rand_indices = np.random.uniform(size=samples) < 0.9
        person_feats[rand_indices, 1] = (np.random.uniform(size=np.count_nonzero(rand_indices)) < 0.25) * 1

        propensities = np.zeros(samples)
        for i, (f1, f2) in enumerate(person_feats):
            choice_set_feature = f1 if cls.choice_set_ignorability else f2
            p_no_bird = (0.75 if choice_set_feature else 0.25) if cls.mnar else 0.75
            if np.random.random() < p_no_bird:
                choice_sets[i, 2] = 0
                propensities[i] = p_no_bird if cls.choice_set_ignorability else (0.5625 if f1 else 0.3125)  # propensities given observation of f1
            else:
                propensities[i] = (1 - p_no_bird) if cls.choice_set_ignorability else (0.4375 if f1 else 0.6875)

        options = np.array([0, 1])
        for i, (f1, f2) in enumerate(person_feats):
            preference_feature = f1 if cls.preference_ignorability else f2
            utils = np.log([3, 1]) if preference_feature else np.log([1, 3])
            choices[i] = np.random.choice(options, p=softmax(utils))

        return choice_sets, choices.astype(int), pd.DataFrame(person_feats[:, 0]), propensities

    @classmethod
    def generate_pytorch(cls, samples, seed):
        choice_sets, choices, person_df, propensities = cls.generate(samples, seed)
        choice_sets = torch.from_numpy(choice_sets)[:, :, None].float()
        choices = torch.tensor(choices)
        return choice_sets, choices, person_df, torch.from_numpy(propensities).float()

    @classmethod
    def load(cls):
        return cls.generate(5000, 0)[:3]


class SyntheticPetsMNAR(SyntheticPetsSuper):
    mnar = True
    choice_set_ignorability = True
    preference_ignorability = True


class SyntheticPetsMCAR(SyntheticPetsSuper):
    mnar = False
    choice_set_ignorability = True
    preference_ignorability = True


class SyntheticPetsCSIgnorableMNAR(SyntheticPetsSuper):
    mnar = True
    choice_set_ignorability = True
    preference_ignorability = False


class SyntheticPetsCSIgnorableMCAR(SyntheticPetsSuper):
    mnar = False
    choice_set_ignorability = True
    preference_ignorability = False


class SyntheticPetsPIgnorableMNAR(SyntheticPetsSuper):
    mnar = True
    choice_set_ignorability = False
    preference_ignorability = True


class SyntheticPetsPIgnorableMCAR(SyntheticPetsSuper):
    mnar = False
    choice_set_ignorability = False
    preference_ignorability = True


class SyntheticPetsNonIgnorableMNAR(SyntheticPetsSuper):
    mnar = True
    choice_set_ignorability = False
    preference_ignorability = False


class SyntheticPetsNonIgnorableMCAR(SyntheticPetsSuper):
    mnar = False
    choice_set_ignorability = False
    preference_ignorability = False


class SyntheticConfoundedCDMSuper(ItemIdentityDataset):
    item_names = list(range(20))
    mnar = None
    choice_set_ignorability = None
    preference_ignorability = None

    @classmethod
    def generate(cls, samples, embedding_dim, seed, context_strength, confounding_strength):
        np.random.seed(seed)

        n_items = len(cls.item_names)

        item_feats = np.random.normal(0, 1, (n_items, embedding_dim))
        item_feats /= np.linalg.norm(item_feats, axis=1, keepdims=True)

        person_feats = np.random.normal(0, 1, (samples, embedding_dim))
        person_feats /= np.linalg.norm(person_feats, axis=1, keepdims=True)

        pulls = np.random.uniform(-context_strength, context_strength, (n_items, n_items))
        np.fill_diagonal(pulls, 0)

        base_utilities = person_feats @ item_feats.T

        propensities = 1 / (1 + np.exp(-base_utilities*confounding_strength))

        # add in some random sets
        row_subset = np.random.choice(np.arange(samples), samples // 4, replace=False)
        propensities[row_subset, :] = 0.5

        choice_sets = np.random.binomial(n=1, p=propensities)

        resample_rows = choice_sets.sum(1) < 2
        while np.count_nonzero(resample_rows) > 0:

            choice_sets[resample_rows] = np.random.binomial(n=1, p=propensities)[resample_rows]
            resample_rows = choice_sets.sum(1) < 2

        utilities = base_utilities + (choice_sets @ pulls)
        utilities[choice_sets == 0] = -np.inf
        choice_probs = softmax(utilities, axis=1)

        options = np.array(cls.item_names)
        choices = np.array([np.random.choice(options, p=choice_probs[i]) for i in range(samples)])

        return choice_sets, choices[:, np.newaxis], pd.DataFrame(person_feats), propensities

    @classmethod
    def generate_pytorch(cls, samples, embedding_dim, seed, context_strength=0.5, confounding_strength=1):
        choice_sets, choices, person_df, propensities = cls.generate(samples, embedding_dim, seed, context_strength, confounding_strength)
        choice_sets = torch.from_numpy(choice_sets)[:, :, None].float()
        choices = torch.tensor(choices)
        return choice_sets, choices, person_df, torch.from_numpy(propensities).float()

    @classmethod
    def load(cls):
        ...


class YoochooseCats(ItemIdentityDataset):
    name = 'yoochoose-cats'

    @classmethod
    def load(cls):
        pickle_fname = f'{DATA_DIR}/pickles/yoochoose-cats.pickle'
        if os.path.isfile(pickle_fname):
            with open(pickle_fname, 'rb') as f:
                return pickle.load(f)

        clicks_df = pd.read_csv(f'{DATA_DIR}/yoochoose-data/yoochoose-clicks.dat', usecols=[0, 2, 3], names=['session_id', 'item_id', 'category'])
        buys_df = pd.read_csv(f'{DATA_DIR}/yoochoose-data/yoochoose-buys.dat', usecols=[0, 2], names=['session_id', 'item_id']).drop_duplicates()

        # filter out sessions where only one category was clicked
        clicks_df = clicks_df[clicks_df.groupby('session_id')['category'].transform('nunique') > 1]
        buys_df = buys_df[buys_df.session_id.isin(clicks_df.session_id.unique())]

        # get a dict of session ids to the categories in them
        choice_set_map = clicks_df.groupby('session_id')['category'].indices

        # add category column to buys_df
        category_map = clicks_df.drop(columns=['session_id']).drop_duplicates('item_id').set_index('item_id')
        buys_df['category'] = buys_df.item_id.map(category_map.category)

        # remove categories with fewer than 100 buys
        buys_df = buys_df[buys_df['category'].map(buys_df['category'].value_counts()) >= 100]

        all_cats = set(buys_df['category'].unique())
        n_cats = len(all_cats)
        cat_index = {cat: i for i, cat in enumerate(sorted(all_cats))}

        choice_sets = []
        choices = []

        for row in tqdm(buys_df.itertuples(index=False), total=len(buys_df)):
            choice_set_clicks = clicks_df.iloc[choice_set_map[row.session_id]]

            choice_set_cats = all_cats.intersection(choice_set_clicks.category)

            choice_set = np.zeros(n_cats, dtype=int)
            choice_set[[cat_index[cat] for cat in choice_set_cats]] = 1
            choice_set[cat_index[row.category]] = 1

            if len(choice_set_cats) >= 2:
                choices.append(cat_index[row.category])
                choice_sets.append(choice_set)

        choice_sets = np.array(choice_sets)
        choices = np.array(choices)[:, np.newaxis]

        # no chooser feats
        person_df = pd.DataFrame(np.zeros(len(choices)))

        with open(pickle_fname, 'wb') as f:
            pickle.dump((choice_sets, choices, person_df), f)

        return choice_sets, choices, person_df


if __name__ == '__main__':
    choice_sets, choices, person_df = YoochooseCats.load_pytorch()
    print(len(choices), len(np.unique(choices)))
