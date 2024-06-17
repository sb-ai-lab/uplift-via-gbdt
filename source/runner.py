import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklift.metrics.metrics import uplift_auc_score


def qini(y_true, uplift, treatment):
    scores = []

    for t_id in range(treatment.max()):
        sl = (treatment == 0) | (treatment == (t_id + 1))
        y_ = y_true[sl]
        upl = uplift[:, t_id][sl]
        trt = np.clip(treatment[sl], 0, 1)

        scores.append(
            uplift_auc_score(y_, upl, trt, )
        )

    return scores


def mse(y_true, uplift, treatment):
    scores = []

    for t_id in range(treatment.max()):
        sl = treatment == (t_id + 1)
        y_ = y_true[sl]
        upl = uplift[:, t_id][sl]

        scores.append(
            ((y_ - upl) ** 2).mean()
        )

    return scores


def prepare_for_cml(data, n_trt=None):
    data = data.copy()
    if n_trt is None:
        n_trt = data['t'].max() + 1

    if 't' in data:
        t = data.pop('t')
        trt = np.array(['control'] + [f'treatment{x}' for x in range(1, n_trt)])
        data['treatment'] = trt[t]

    if 'p' in data:
        p = data.pop('p')
        data['p'] = {f'treatment{x}': p[:, x - 1] for x in range(1, n_trt)}

    return data


def train_meta_uber(train, test, params, factory):
    train, test = train.copy(), test.copy()
    nt = train['t'].max() + 1

    oof_pred = np.zeros((train['X'].shape[0], nt - 1), dtype=np.float32)
    test_pred = np.zeros((test['X'].shape[0], nt - 1), dtype=np.float32)

    folds = StratifiedKFold(5, shuffle=True, random_state=42)

    scores = {
        'valid_ext': [],
        'test_ext': [],
    }

    if 'effect' in train:
        scores['valid_mse_ext'] = []
        scores['test_mse_ext'] = []

    strf = train['t'] + nt * train['y']
    for n, (f0, f1) in enumerate(folds.split(strf, strf)):
        X_tr, X_val = train['X'][f0], train['X'][f1]
        t_tr, t_val = train['t'][f0], train['t'][f1]
        y_tr, y_val = train['y'][f0], train['y'][f1]
        p_tr, p_val = train['p'][f0], train['p'][f1]
        model = factory(params)

        _ = model.fit_predict(**prepare_for_cml(
            {'X': X_tr, 'y': y_tr, 't': t_tr, 'p': p_tr}, n_trt=nt
        ), return_ci=False)

        # baseline
        oof_pred[f1] = model.predict(**prepare_for_cml(
            {'X': X_val, 'p': p_val}, n_trt=nt
        ))
        tt = model.predict(**prepare_for_cml(
            {'X': test['X'], 'p': test['p']}, n_trt=nt
        ))

        scores['test_ext'].append(qini(test['y'], tt, test['t']))
        scores['valid_ext'].append(qini(train['y'][f1], oof_pred[f1], train['t'][f1]))

        # if effect ...
        if 'effect' in train:
            scores['test_mse_ext'].append(mse(test['effect'], tt, test['t']))
            scores['valid_mse_ext'].append(mse(train['effect'][f1], oof_pred[f1], train['t'][f1]))

        test_pred += tt

    test_pred /= 5

    scores = {
        **scores,
        'valid': qini(train['y'], oof_pred, train['t']),
        'test': qini(test['y'], test_pred, test['t']),

    }

    if 'effect' in train:
        scores = {
            **scores,
            'valid_mse': mse(train['effect'], oof_pred, train['t']),
            'test_mse': mse(test['effect'], test_pred, test['t']),

        }

    return scores, oof_pred, test_pred


def train_crf(train, test, params, factory):
    train, test = train.copy(), test.copy()
    nt = train['t'].max() + 1

    oof_pred = np.zeros((train['X'].shape[0], nt - 1), dtype=np.float32)
    test_pred = np.zeros((test['X'].shape[0], nt - 1), dtype=np.float32)

    folds = StratifiedKFold(5, shuffle=True, random_state=42)

    scores = {
        'valid_ext': [],
        'test_ext': [],
    }

    if 'effect' in train:
        scores['valid_mse_ext'] = []
        scores['test_mse_ext'] = []

    strf = train['t'] + nt * train['y']
    treat = 1 - np.isnan(train['trg'][:, 1:])

    for n, (f0, f1) in enumerate(folds.split(strf, strf)):
        X_tr, X_val = train['X'][f0], train['X'][f1]
        t_tr, t_val = treat[f0], treat[f1]
        y_tr, y_val = train['y'][f0], train['y'][f1]
        model = factory(params)

        model.fit(X_tr, t_tr, y_tr)

        # baseline
        oof_pred[f1] = model.predict(X_val)
        tt = model.predict(test['X'])

        scores['test_ext'].append(qini(test['y'], tt, test['t']))
        scores['valid_ext'].append(qini(train['y'][f1], oof_pred[f1], train['t'][f1]))

        # if effect ...
        if 'effect' in train:
            scores['test_mse_ext'].append(mse(test['effect'], tt, test['t']))
            scores['valid_mse_ext'].append(mse(train['effect'][f1], oof_pred[f1], train['t'][f1]))

        test_pred += tt

    test_pred /= 5

    scores = {
        **scores,
        'valid': qini(train['y'], oof_pred, train['t']),
        'test': qini(test['y'], test_pred, test['t']),

    }

    if 'effect' in train:
        scores = {
            **scores,
            'valid_mse': mse(train['effect'], oof_pred, train['t']),
            'test_mse': mse(test['effect'], test_pred, test['t']),

        }

    return scores, oof_pred, test_pred


def train_drnet(train, test, params, factory):
    train, test = train.copy(), test.copy()
    nt = train['t'].max() + 1

    oof_pred = np.zeros((train['X'].shape[0], nt - 1), dtype=np.float32)
    test_pred = np.zeros((test['X'].shape[0], nt - 1), dtype=np.float32)

    folds = StratifiedKFold(5, shuffle=True, random_state=42)

    scores = {
        'valid_ext': [],
        'test_ext': [],
    }

    if 'effect' in train:
        scores['valid_mse_ext'] = []
        scores['test_mse_ext'] = []

    strf = train['t'] + nt * train['y']
    treat = 1 - np.isnan(train['trg'][:, 1:])

    for n, (f0, f1) in enumerate(folds.split(strf, strf)):

        X_tr, X_val = train['X'][f0], train['X'][f1]
        t_tr, t_val = treat[f0], treat[f1]
        y_tr, y_val = train['y'][f0], train['y'][f1]

        tt = np.zeros((test['X'].shape[0], t_tr.shape[1]), dtype=np.float32)

        for t in range(treat.shape[1]):
            model = factory(params)
            sl_tr = (t_tr.sum(axis=1) == 0) | (t_tr[:, t] == 1)
            sl_val = (t_val.sum(axis=1) == 0) | (t_val[:, t] == 1)

            model.fit(X_tr[sl_tr], y_tr[sl_tr], t_tr[sl_tr][:, t], X_val[sl_val], y_val[sl_val], t_val[sl_val][:, t])
            oof_pred[f1, t] = model.predict(X_val)
            tt[:, t] = model.predict(test['X'])

        scores['test_ext'].append(qini(test['y'], tt, test['t']))
        scores['valid_ext'].append(qini(train['y'][f1], oof_pred[f1], train['t'][f1]))

        # if effect ...
        if 'effect' in train:
            scores['test_mse_ext'].append(mse(test['effect'], tt, test['t']))
            scores['valid_mse_ext'].append(mse(train['effect'][f1], oof_pred[f1], train['t'][f1]))

        test_pred += tt

    test_pred /= 5

    scores = {
        **scores,
        'valid': qini(train['y'], oof_pred, train['t']),
        'test': qini(test['y'], test_pred, test['t']),

    }

    if 'effect' in train:
        scores = {
            **scores,
            'valid_mse': mse(train['effect'], oof_pred, train['t']),
            'test_mse': mse(test['effect'], test_pred, test['t']),

        }

    return scores, oof_pred, test_pred


def train_pb_upd_weight(train, test, params, factory):
    train, test = train.copy(), test.copy()
    nt = train['t'].max() + 1

    for data in [train, test]:
        t = data['t']

        trg = np.zeros((t.shape[0], nt), dtype=np.float32)
        trg[:] = np.nan
        rows, cols = np.nonzero(t[:, np.newaxis] == np.arange(nt)[np.newaxis, :])
        trg[rows, cols] = data['y']

        data['new_y'] = trg

    oof_pb = [np.zeros((train['X'].shape[0], nt - 1), dtype=np.float32) for _ in range(11)]
    test_pb = [np.zeros((test['X'].shape[0], nt - 1), dtype=np.float32) for _ in range(11)]

    folds = StratifiedKFold(5, shuffle=True, random_state=42)

    scores = {

        'valid_ext': [],
        'test_ext': [],
        'valid_ext_w': [],
        'test_ext_w': []

    }

    if 'effect' in train:
        scores['valid_mse_ext'] = []
        scores['test_mse_ext'] = []
        scores['valid_mse_ext_w'] = []
        scores['test_mse_ext_w'] = []

    models = []

    strf = train['t'] + nt * train['y']

    for n, (f0, f1) in enumerate(folds.split(strf, strf)):

        X_tr, X_val = train['X'][f0], train['X'][f1]
        y_tr, y_val = train['new_y'][f0], train['new_y'][f1]

        model = factory(params)

        model.fit(X_tr, y_tr, eval_sets=[{'X': X_val, 'y': y_val}])
        models.append(model)

        idx = np.searchsorted(np.linspace(0, 1, 11), model.loss.weight)

        # baseline
        # get valid scores and test scores for each w
        vs, ts = [], []
        vm, tm = [], []

        for k, w in enumerate(np.linspace(0, 1, 11)):
            model.loss.weight = w

            oof_pb[k][f1] = model.predict(X_val, batch_size=1e10)
            tt = model.predict(test['X'], batch_size=1e10)
            test_pb[k] += tt

            vs.append(
                qini(train['y'][f1], oof_pb[k][f1], train['t'][f1])
            )

            ts.append(
                qini(test['y'], tt, test['t'])
            )

            if 'effect' in train:
                vm.append(
                    mse(train['effect'][f1], oof_pb[k][f1], train['t'][f1])
                )

                tm.append(
                    mse(test['effect'], tt, test['t'])
                )

        scores['valid_ext_w'].append(vs)
        scores['test_ext_w'].append(ts)

        if 'effect' in train:
            scores['valid_mse_ext_w'].append(vm)
            scores['test_mse_ext_w'].append(tm)

    scores['valid_ext_w'] = np.array(scores['valid_ext_w'])
    scores['test_ext_w'] = np.array(scores['test_ext_w'])

    # print(scores['valid_ext_w'].shape)
    scores['valid_w'] = [
        qini(train['y'], oof_pb[x], train['t']) for x in range(11)

    ]

    scores['test_w'] = [
        qini(test['y'], test_pb[x], test['t']) for x in range(11)

    ]

    scores['best_k'] = np.argmax(np.mean(scores['valid_w'], axis=1))

    scores['valid_ext'] = scores['valid_ext_w'][:, idx]
    scores['test_ext'] = scores['test_ext_w'][:, idx]

    if 'effect' in train:
        scores['valid_mse_ext_w'] = np.array(scores['valid_mse_ext_w'])
        scores['test_mse_ext_w'] = np.array(scores['test_mse_ext_w'])

        scores['valid_mse_w'] = [
            mse(train['effect'], oof_pb[x], train['t']) for x in range(11)

        ]

        scores['test_mse_w'] = [
            mse(test['effect'], test_pb[x], test['t']) for x in range(11)

        ]

        scores['valid_mse_ext'] = scores['valid_mse_ext_w'][:, idx]
        scores['test_mse_ext'] = scores['test_mse_ext_w'][:, idx]

    for tt in test_pb:
        tt /= 5

    scores = {
        **scores,
        'valid': scores['valid_w'][scores['best_k']],
        'test': scores['test_w'][scores['best_k']],
    }

    if 'effect' in train:
        scores = {
            **scores,
            'valid_mse': scores['valid_mse_w'][scores['best_k']],
            'test_mse': scores['test_mse_w'][scores['best_k']],
        }

    return scores, oof_pb, test_pb
