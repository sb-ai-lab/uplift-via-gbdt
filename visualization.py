import joblib
import numpy as np
import pandas as pd


def get_mse(ds, pred):
    """

    :param ds:
    :param pred:
    :return:
    """
    mse = []
    mse_stds = []
    for t in range(1, ds['t'].max() + 1):
        stds = []

        sl = (ds['t'] == t)
        y = ds['effect'][sl]
        p = pred[sl][:, t - 1]
        mse.append(((y - p) ** 2).mean())

        for _ in range(100):
            idx = np.random.randint(0, p.shape[0], size=p.shape[0])
            stds.append(((y[idx] - p[idx]) ** 2).mean())

        mse_stds.append(np.std(stds, ddof=1))

    return mse, mse_stds


def get_results_simple(
        data, experiment, task, nstart=0, nstop=5, key='test',
        selector_fn=lambda x: np.mean(x.values), weight=None
):
    """

    :param data:
    :param experiment:
    :param task:
    :param nstart:
    :param nstop:
    :param key:
    :param selector_fn:
    :param weight:
    :return:
    """
    res = {'AUUC': {}, 'MSE': {}, 'ATE': {}, 'ATE_ERR, %': {}, 'QINI': {}}

    auuc = []
    auuc_std = []
    mses = []
    mse_stds = []

    for n in range(nstart, nstop):
        # dataset for metrics'
        data_ = data.split('/')[-1]
        ds_name = key if key == 'test' else 'trian'
        ds = joblib.load(f'datasets/{data}_{n}/{ds_name}.pkl')

        # get best trial
        study = joblib.load(f'{experiment}/{data_}_{n}/{task}/study.pkl')
        best = max(study.trials, key=selector_fn)
        # print(best.params)
        # get all scores
        scores = joblib.load(
            f'{experiment}/{data_}_{n}/{task}/trial_{best.number}/scores.pkl'
        )
        pred = joblib.load(f'{experiment}/{data_}_{n}/{task}/trial_{best.number}/{key}_pred.pkl')

        if weight is not None:
            idx = np.searchsorted(np.linspace(0, 1, 11), weight)
            sc = scores[f'{key}_ext_w'][:, idx, :]
            pred = pred[idx]
        else:
            sc = scores[f'{key}_ext']
            if len(pred) == 11:
                pred = pred[5]

        # calc all the additional metrics we need
        # loop by treatments

        if 'effect' in ds:
            mse_, mse_stds_ = get_mse(ds, pred)
            mses.append(np.array(mse_))
            mse_stds.append(np.array(mse_stds_))

        auuc.append(np.mean(sc, axis=0))
        auuc_std.append(np.std(sc, axis=0, ddof=1))

    # save auucs
    if 'effect' in ds:
        res['MSE']['mean'] = np.mean(mses, axis=0)
        res['MSE']['std'] = np.mean(mse_stds, axis=0)

    res['AUUC']['mean'] = np.mean(auuc, axis=0)
    res['AUUC']['std'] = np.std(auuc_std, axis=0)

    return res


def get_baselines_results(experiment, datasets, models, K=5):
    """

    :param experiment:
    :param datasets:
    :param models:
    :param K:
    :return:
    """
    res = []

    for dataset in datasets:
        for model in models:

            D = get_results_simple(
                dataset,
                experiment,
                model,
                key='test',
                nstop=K
            )

            for key in ['mean', 'std']:
                df = pd.DataFrame({x: D[x][key] for x in D if key in D[x]}, )
                df['treat'] = np.arange(df.shape[0])
                df['data'] = dataset
                df['model'] = model
                df['stat'] = key

                res.append(df)

    return res


def get_pb_results(experiment, datasets, weights, K=5):
    """

    :param experiment:
    :param datasets:
    :param weights:
    :param K:
    :return:
    """
    res = []

    for dataset in datasets:
        for w in weights:
            D = get_results_simple(
                dataset,
                experiment,
                'pb-pb-pb_lc_f_t',
                key='test',
                nstop=K,
                weight=round(w, 1)
            )
            for key in ['mean', 'std']:
                df = pd.DataFrame({x: D[x][key] for x in D if key in D[x]}, )
                df['treat'] = np.arange(df.shape[0])
                df['data'] = dataset
                df['model'] = 'pb-pb-pb_lc_f_t' + str(w)
                df['stat'] = key

                res.append(df)
    return res


def get_pb_weighted_results(experiment, dataset, K=5):
    """

    :param experiment:
    :param dataset:
    :param K:
    :return:
    """
    res = []

    for w in np.linspace(0, 1, 11):
        w = round(w, 1)
        D = get_results_simple(
            dataset,
            experiment,
            'pb-pb-pb_lc_f_t',
            key='test',
            nstop=K,
            weight=w
        )
        for key in ['mean', 'std']:
            df = pd.DataFrame({x: D[x][key] for x in D if key in D[x]}, )
            df['treat'] = np.arange(df.shape[0])
            df['data'] = dataset
            df['model'] = 'pb-pb-pb_lc_f_t' + str(w)
            df['stat'] = key

            res.append(df)

    res = pd.concat(res)
    res['w'] = res['model'].map(lambda x: x[-3:]).astype(float)
    return res


def replace_index(df, mapping):
    df = df.loc[list(mapping.keys())]
    df = df.reset_index()
    df['model'] = df['model'].map(mapping)
    df = df.set_index('model')

    return df


def get_datasets_summary(res, stat, mapping=None, round_mean=3, round_std=4):
    """

    :param res:
    :param stat:
    :param round_mean:
    :param round_std:
    :return:
    """
    df_mean = res.query('stat == "mean"')[[stat, 'treat', 'data', 'model']]
    df_mean = pd.pivot_table(
        df_mean, values=stat, index='model', columns=['data', 'treat']

    )

    df_std = res.query('stat == "std"')[[stat, 'treat', 'data', 'model']]
    df_std = pd.pivot_table(
        df_std, values=stat, index='model', columns=['data', 'treat']

    )
    tot = df_mean.round(round_mean).astype(str) + '\u00b1' + df_std.round(round_std).astype(str)
    if mapping is not None:
        tot = replace_index(tot, mapping)

    return tot


def get_rank_stats(res, mapping):
    avg_rank = res.query('stat == "mean"').copy()
    avg_rank['AUUC_rank'] = avg_rank.groupby(['data', 'treat'])['AUUC'].rank(method='dense', ascending=False)
    avg_rank['AUUC_from_top'] = 1 - avg_rank['AUUC'] / avg_rank.groupby(['data', 'treat'])['AUUC'].transform('max')
    avg_rank['AUUC_from_top'] = avg_rank['AUUC_from_top'] * 100

    avg_rank['MSE_rank'] = avg_rank.groupby(['data', 'treat'])['MSE'].rank(method='dense', ascending=True)
    avg_rank['MSE_from_top'] = avg_rank['MSE'] / avg_rank.groupby(['data', 'treat'])['MSE'].transform('min') - 1
    avg_rank['MSE_from_top'] = avg_rank['MSE_from_top'] * 100

    avg_rank = avg_rank.groupby('model')[['AUUC_rank', 'MSE_rank', 'AUUC_from_top', 'MSE_from_top']] \
        .mean().round(1)

    if mapping is not None:
        avg_rank = replace_index(avg_rank, mapping)

    return avg_rank


def to_latex(df, direction, is_str=True):
    """

    :param df:
    :param direction:
    :return:
    """
    df = df.copy()

    best = []

    for col in df.columns:
        ser = df[col]
        if is_str:
            ser = ser.str.split('\u00b1').map(lambda x: x[0])
        ser = ser.astype(float)

        best.append(ser.max() if direction == 'max' else ser.min())

    for n, col in enumerate(df.columns):
        ser = df[col]
        if is_str:
            ser = ser.str.split('\u00b1').map(lambda x: x[0])
        ser = ser.astype(float)
        sl = ser == best[n]
        df[col] = df[col].astype(str)
        df[col].loc[sl] = "\\textbf{" + df[col].loc[sl].astype(str) + "}"

    df = df.reset_index()

    df['model'] = "\\textbf{" + df['model'] + "}"

    df = df.apply(lambda x: ' & '.join(x), axis=1).tolist()
    df = ' \\\\ \n'.join(df)

    return df
