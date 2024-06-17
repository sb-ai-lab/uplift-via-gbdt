import os

import joblib
import numpy as np
import pandas as pd
from causalml.dataset import make_uplift_classification
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklift import datasets
from xgboost import XGBClassifier

DATASET_PATH = os.path.dirname(os.path.abspath(__file__))


def make_testset(X, y, t, fold, random_state=42, stratify=True, effect=False):
    train, test = {}, {}
    nt = t.max() + 1

    trg = np.zeros((t.shape[0], nt), dtype=np.float32)
    trg[:] = np.nan
    rows, cols = np.nonzero(t[:, np.newaxis] == np.arange(nt)[np.newaxis, :])
    trg[rows, cols] = y

    if stratify:
        stratify = t + nt * y
        folds = StratifiedKFold(5, random_state=random_state, shuffle=True).split(stratify, stratify)
    else:
        folds = KFold(5, random_state=random_state, shuffle=True).split(y, y)

    for n, (f0, f1) in enumerate(folds):

        if n == fold:
            break

    train['X'], test['X'] = X[f0], X[f1]
    train['y'], test['y'] = y[f0], y[f1]
    train['t'], test['t'] = t[f0], t[f1]
    train['trg'], test['trg'] = trg[f0], trg[f1]

    if effect:
        for data in [train, test]:
            data['X'], data['effect'] = data['X'][:, :-1], data['X'][:, -1]

    return train, test


def save_data(train, test, alias, i):
    folder = os.path.join(DATASET_PATH, f'{alias}_{i}')
    os.makedirs(folder, exist_ok=True)
    joblib.dump(train, os.path.join(folder, 'train.pkl'))
    joblib.dump(test, os.path.join(folder, 'test.pkl'))

    return


def get_synth_6_trt():
    names = ["control", "treatment1", "treatment2", "treatment3", "treatment4", "treatment5", "treatment6"]

    df, x_names = make_uplift_classification(
        n_samples=10000,
        treatment_name=names,
        y_name="conversion",
        n_classification_features=100,
        n_classification_informative=20,
        n_classification_redundant=10,
        n_classification_repeated=10,
        n_uplift_increase_dict={
            "treatment1": 3, "treatment2": 5, "treatment3": 7, "treatment4": 9, "treatment5": 11, "treatment6": 13,
        },
        n_uplift_decrease_dict={
            "treatment1": 1, "treatment2": 2, "treatment3": 3, "treatment4": 3, "treatment5": 4, "treatment6": 4,
        },
        delta_uplift_increase_dict={
            "treatment1": 0.05,
            "treatment2": 0.1,
            "treatment3": 0.12,
            "treatment4": 0.15,
            "treatment5": 0.17,
            "treatment6": 0.2,
        },
        delta_uplift_decrease_dict={
            "treatment1": 0.01,
            "treatment2": 0.02,
            "treatment3": 0.03,
            "treatment4": 0.05,
            "treatment5": 0.06,
            "treatment6": 0.07,
        },
        n_uplift_increase_mix_informative_dict={
            "treatment1": 1,
            "treatment2": 2,
            "treatment3": 3,
            "treatment4": 4,
            "treatment5": 5,
            "treatment6": 6,
        },
        n_uplift_decrease_mix_informative_dict={
            "treatment1": 1,
            "treatment2": 1,
            "treatment3": 1,
            "treatment4": 1,
            "treatment5": 1,
            "treatment6": 1,
        },
        positive_class_proportion=0.2,
        random_seed=42,
    )

    tkeys = {f"treatment{x}": x for x in range(1, 7)}
    tkeys['control'] = 0

    alias = 'synth1'

    X = df.drop(['treatment_group_key', 'conversion', ], axis=1).values.astype(np.float32)
    y = df['conversion'].values.astype(np.float32)
    t = df['treatment_group_key'].map(tkeys).values.astype(np.int32)

    for i in range(5):
        train, test = make_testset(X, y, t, fold=i, random_state=42, stratify=True, effect=True)
        save_data(train, test, alias, i)

    return


def get_hillstom():
    alias = 'hillstrom'
    data = datasets.fetch_hillstrom()
    data, target, treatment = data['data'], data['target'], data['treatment']

    data['history_segment'] = data['history_segment'].str.slice(0, 1).astype(int)
    for col in ['zip_code', 'channel']:
        data[col] = pd.factorize(data[col])[0]

    X = data.values.astype(np.float32)
    y = target.values.astype(np.float32)
    t = treatment.map({'No E-Mail': 0, 'Mens E-Mail': 1, 'Womens E-Mail': 2}).values.astype(np.int32)

    for i in range(5):
        train, test = make_testset(X, y, t, fold=i, random_state=42, stratify=False, effect=False)
        save_data(train, test, alias, i)

    return


def get_criteo():
    alias = 'criteo'
    data = datasets.fetch_criteo()
    data, target, treatment = data['data'], data['target'], data['treatment']
    t = treatment.values.astype(np.int32)
    y = target.values.astype(np.float32)

    np.random.seed(42)

    idx = np.arange(t.shape[0])
    idx0, idx1 = idx[y == 0], idx[y == 1]
    np.random.shuffle(idx0)

    idx0 = idx0[:1000000]

    idx = np.concatenate([idx0, idx1])
    np.random.shuffle(idx)

    X = data.values.astype(np.float32)[idx]
    y = y[idx]
    t = t[idx]

    for i in range(5):
        train, test = make_testset(X, y, t, fold=i, random_state=42, stratify=False, effect=False)
        save_data(train, test, alias, i)

    return


def get_lenta():
    alias = 'lenta'
    data = datasets.fetch_lenta()

    data, target, treatment = data['data'], data['target'], data['treatment']
    t = treatment.map({'control': 0, 'test': 1}).values.astype(np.int32)
    y = target.values.astype(np.float32)

    data['gender'] = (data['gender'] == data['gender'].iloc[0])
    # for simplicity - fill NaNs with median
    data = data.fillna(data.median())
    X = data.values.astype(np.float32)

    for i in range(5):
        train, test = make_testset(X, y, t, fold=i, random_state=42, stratify=False, effect=False)
        save_data(train, test, alias, i)

    return


def get_megafon():
    alias = 'megafon'
    data = datasets.fetch_megafon()

    data, target, treatment = data['data'], data['target'], data['treatment']
    t = treatment.map({'control': 0, 'treatment': 1}).values.astype(np.int32)
    y = target.values.astype(np.float32)
    X = data.values.astype(np.float32)

    for i in range(5):
        train, test = make_testset(X, y, t, fold=i, random_state=42, stratify=False, effect=False)
        save_data(train, test, alias, i)

    return


# params for propensity estimator
params_xgb = {
    'n_estimators': 1000,
    'learning_rate': 0.01,
    'max_depth': 3,
    'tree_method': 'gpu_hist',
    'gpu_id': 0,
    'min_child_weight': 0,
    'lambda': 1,
    'max_bin': 256,
    'gamma': 0,
    'alpha': 0,
}


def get_trt_slice(x, y, t):
    sl = np.isin(y, [0, t])
    y = (y[sl] == t).astype(np.float32)
    x = x[sl]

    return x, y


def set_propensity(train, test, params, cutoff=0.55):
    n_trt = train['t'].max()

    X, y = train['X'], train['t']
    X_test = test['X']

    folds = KFold(5, shuffle=True, random_state=42)

    oof_pred = np.zeros((X.shape[0], n_trt), dtype=np.float32)
    test_pred = np.zeros((X_test.shape[0], n_trt), dtype=np.float32)

    scores = []
    priors = []

    for n, (f0, f1) in enumerate(folds.split(y, y)):

        x_tr, x_val = X[f0], X[f1]
        y_tr, y_val = y[f0], y[f1]
        score = []
        prior = []
        for i in range(n_trt):
            # get target slice for control + curr treatment
            ds = [
                get_trt_slice(x, y, i + 1) for (x, y) in
                [[x_tr, y_tr], [x_val, y_val]]
            ]

            model = XGBClassifier(**params)
            model.fit(
                *ds[0], eval_set=[ds[1]],
                early_stopping_rounds=100, eval_metric='auc', verbose=1000
            )

            score.append(
                roc_auc_score(ds[1][1], model.predict_proba(ds[1][0])[:, 1])
            )
            prior.append(ds[0][1].mean())

            oof_pred[f1, i] = model.predict_proba(x_val)[:, 1]
            test_pred[:, i] += model.predict_proba(X_test)[:, 1]

        scores.append(score)
        priors.append(prior)

    scores = np.array(scores).mean(axis=0)
    priors = np.array(priors).mean(axis=0)
    test_pred /= 5

    for i in range(n_trt):
        if scores[i] < cutoff:
            oof_pred[:, i] = priors[i]
            test_pred[:, i] = priors[i]

    train['p'] = oof_pred
    test['p'] = test_pred

    return train, test


if __name__ == '__main__':
    print('Fetching the data...')
    # fetch datasets
    get_synth_6_trt()
    get_hillstom()
    get_criteo()
    get_megafon()
    get_lenta()

    print('Estimating propensities...')
    # create propensity scores - required for some models
    for alias in ['synth1', 'hillstrom', 'criteo', 'megafon', 'lenta']:
        for i in range(5):
            folder = os.path.join(DATASET_PATH, f'{alias}_{i}')
            train = joblib.load(os.path.join(folder, 'train.pkl'))
            test = joblib.load(os.path.join(folder, 'test.pkl'))

            train, test = set_propensity(train, test, params_xgb, cutoff=0.55)
            save_data(train, test, alias, i)

    print('Done')
