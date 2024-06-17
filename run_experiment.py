import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-p', '--path', type=str)  # dataset path
parser.add_argument('-n', '--njobs', type=int, default=8)
parser.add_argument('-s', '--seed', type=int, default=42)
parser.add_argument('-d', '--device', type=str, default='0')
parser.add_argument('-r', '--runner', type=str)  # train_fn_dict
parser.add_argument('-t', '--tuner', type=str)  # obj_dict
parser.add_argument('-m', '--model', type=str)  # model_dict
parser.add_argument('-c', '--config', type=str)  # general confit file

if __name__ == '__main__':

    import os

    args = parser.parse_args()
    str_nthr = str(args.njobs)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    os.environ["OMP_NUM_THREADS"] = str_nthr  # export OMP_NUM_THREADS=4
    os.environ["OPENBLAS_NUM_THREADS"] = str_nthr  # export OPENBLAS_NUM_THREADS=4
    os.environ["MKL_NUM_THREADS"] = str_nthr  # export MKL_NUM_THREADS=6
    os.environ["VECLIB_MAXIMUM_THREADS"] = str_nthr  # export VECLIB_MAXIMUM_THREADS=4
    os.environ["NUMEXPR_NUM_THREADS"] = str_nthr  # export NUMEXPR_NUM_THREADS=6

    import optuna
    import numpy as np
    import shutil
    import yaml
    from copy import deepcopy
    from source.objective import *
    from source.factory import *
    from source.runner import *

    np.random.seed(args.seed)

    train_fn_dict = {
        'meta': train_meta_uber,
        'pb': train_pb_upd_weight,
        'crf': train_crf,
        'dr': train_drnet,

    }

    obj_dict = {
        'xgb_single': ObjectiveSingleXGB,  # xgboost with single tuning
        'pb': ObjectivePB,  # our model
        'crf': ObjectiveCRF,
        'dr': ObjectiveDR,
        'dcn': ObjectiveDCN
    }

    model_dict = {
        'xgb_t': (
            get_tlearner_xgb, ['treatment_learner', 'control_learner']
        ),

        'xgb_x': (
            get_xlearner_xgb,
            [
                'control_outcome_learner',
                'treatment_outcome_learner',
                'control_effect_learner',
                'treatment_effect_learner'
            ]
        ),

        'xgb_r': (
            get_rlearner_xgb,
            ['outcome_learner', 'effect_learner']
        ),

        'xgb_dr': (
            get_drlearner_xgb,
            ['control_outcome_learner', 'treatment_outcome_learner', 'treatment_effect_learner']
        ),

        # py-boost learner
        'pb_lc_f_t': (
            lambda x: get_pb_uplift(x, False, True),
            ['params']
        ),

        'crf': (
            get_crf, ['params']
        ),
        'dr': (
            lambda x: get_drnet(x, cat_cols=[5, 7] if 'hillstrom' in args.path else None),
            ['params']
        ),
        'dcn': (
            lambda x: get_dcn(x, cat_cols=[5, 7] if 'hillstrom' in args.path else None),
            ['params']
        )
    }

    train_fn = train_fn_dict[args.runner]
    Obj = obj_dict[args.tuner]
    factory, keys = model_dict[args.model]

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    os.makedirs(config['experiment'], exist_ok=True)

    params_key = args.model.split('_')[0]
    params = deepcopy(config[params_key])

    ds_name = args.path
    if ds_name[-1] == '/':
        ds_name = ds_name[:-1]
    study_path = os.path.join(config['experiment'], os.path.basename(ds_name),
                              f'{args.runner}-{args.tuner}-{args.model}')
    # remove previous runs
    try:
        shutil.rmtree(study_path)
    except FileNotFoundError:
        pass

    # get data
    train = joblib.load(os.path.join(args.path, 'train.pkl'))
    test = joblib.load(os.path.join(args.path, 'test.pkl'))
    print('Study started')
    # run study
    study = optuna.create_study(
        directions=['maximize'] * (train['t'].max()),
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=config['optuna']['n_startup_trials'],
            multivariate=config['optuna']['multivariate']
        )

    )
    objective = Obj(
        # dataset and learner
        train, test, train_fn,
        # facroty,
        factory=factory,
        # keys
        keys=keys,
        # params
        params=params,
        folder=study_path
    )

    study.optimize(objective, n_trials=config['optuna']['n_trials'], timeout=config['optuna']['timeout'])
    joblib.dump(study, os.path.join(study_path, 'study.pkl'))
