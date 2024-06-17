import os
from copy import deepcopy

import joblib
import numpy as np


class Objective:

    def __init__(self, train, test, runner, factory, params, folder, keys=('params',)):

        # data and method
        self.train = train
        self.test = test
        self.runner = runner
        # how to create model
        self.factory = factory
        # keys for each meta learner base model
        self.keys = keys
        # params
        if params is None:
            params = {}
        self.params = params
        # where to save
        self.folder = folder
        os.makedirs(folder, exist_ok=True)

    def _set_params(self, trial, params):

        total_params = {}
        params = self.set_params(trial, deepcopy(params))

        for key in self.keys:
            total_params[key] = params

        return total_params

    def __call__(self, trial):

        params = deepcopy(self.params)
        trial_name = f'trial_{trial.number}'
        folder = os.path.join(self.folder, trial_name)
        os.makedirs(folder, exist_ok=True)
        params = self._set_params(trial, params)

        scores, oof_pred, test_pred = self.runner(self.train, self.test, params, self.factory)
        joblib.dump(scores, os.path.join(folder, 'scores.pkl'))
        joblib.dump(params, os.path.join(folder, 'params.pkl'))
        joblib.dump(oof_pred, os.path.join(folder, 'oof_pred.pkl'))
        joblib.dump(test_pred, os.path.join(folder, 'test_pred.pkl'))

        print(len(scores['valid']))
        print(np.mean(scores['test_ext'], axis=0))

        return list(np.array(scores['valid_ext']).mean(axis=0))  # scores['valid'] # np.mean(scores['valid'])


class ObjectiveMulti(Objective):

    def _set_params(self, trial, params):
        total_params = {}
        for key in self.keys:
            total_params[key] = self.set_params(trial, deepcopy(params), key)

        return total_params


class ObjectiveSingleXGB(Objective):

    def set_params(self, trial, params):
        params['min_child_weight'] = trial.suggest_float("min_child_weight", 1e-5, 10, log=True)
        params['subsample'] = trial.suggest_float("subsample", 0.7, 1.0)
        params['colsample_bytree'] = trial.suggest_float("colsample_bytree", 0.7, 1.0)
        params['max_depth'] = trial.suggest_int("max_depth", 2, 6)
        params['lambda'] = trial.suggest_float("lambda", .1, 50, log=True)
        # params['learning_rate'] = trial.suggest_float("learning_rate", .01, 0.3, log=True)
        params['learning_rate'] = trial.suggest_float("learning_rate", .005, 0.1, log=False)
        params['gamma'] = trial.suggest_float("gamma", 1e-5, 100., log=True)

        return params


class ObjectiveMultiXGB(ObjectiveMulti):

    def set_params(self, trial, params, prefix):
        params['min_child_weight'] = trial.suggest_float(f"{prefix}_min_child_weight", 1e-5, 10, log=True)
        params['subsample'] = trial.suggest_float(f"{prefix}_subsample", 0.7, 1.0)
        params['colsample_bytree'] = trial.suggest_float(f"{prefix}_colsample_bytree", 0.7, 1.0)
        params['max_depth'] = trial.suggest_int(f"{prefix}_max_depth", 2, 6)
        params['lambda'] = trial.suggest_float(f"{prefix}_lambda", .1, 50, log=True)
        # params['learning_rate'] = trial.suggest_float(f"{prefix}_learning_rate", .01, 0.3, log=True)
        params['learning_rate'] = trial.suggest_float(f"{prefix}_learning_rate", .005, 0.1, log=False)
        params['gamma'] = trial.suggest_float(f"{prefix}_gamma", 1e-5, 100., log=True)

        return params


class ObjectivePB(Objective):

    def set_params(self, trial, params):
        params['min_data_in_leaf'] = trial.suggest_int("min_data_in_leaf", 1, 100, log=True)
        params['subsample'] = trial.suggest_float("subsample", 0.7, 1.0)
        params['colsample'] = trial.suggest_float("colsample", 0.7, 1.0)
        params['max_depth'] = trial.suggest_int("max_depth", 2, 6)
        params['lambda_l2'] = trial.suggest_float("lambda_l2", .1, 50, log=True)
        # params['lr'] = trial.suggest_float("lr", .01, 0.3, log=True)
        params['lr'] = trial.suggest_float("lr", .005, 0.1, log=False)
        params['min_gain_to_split'] = trial.suggest_float("min_gain_to_split", 1e-5, 100., log=True)

        #         if 'weight' in params:
        #             params['weight'] = trial.suggest_float("weight", 0, 1, log=False)

        return params


class ObjectiveMultiPB(ObjectiveMulti):

    def set_params(self, trial, params, prefix):
        params['min_data_in_leaf'] = trial.suggest_int(f"{prefix}_min_data_in_leaf", 1, 100, log=True)
        params['subsample'] = trial.suggest_float(f"{prefix}_subsample", 0.7, 1.0)
        params['colsample'] = trial.suggest_float(f"{prefix}_colsample", 0.7, 1.0)
        params['max_depth'] = trial.suggest_int(f"{prefix}_max_depth", 2, 6)
        params['lambda_l2'] = trial.suggest_float(f"{prefix}_lambda_l2", .1, 50, log=True)
        params['lr'] = trial.suggest_float(f"{prefix}_lr", .005, 0.1, log=False)
        params['min_gain_to_split'] = trial.suggest_float(f"{prefix}_min_gain_to_split", 1e-5, 100., log=True)

        return params


class ObjectiveLGB(Objective):

    def set_params(self, trial, params):
        params['min_child_samples'] = trial.suggest_int("min_data_in_leaf", 1, 100, log=True)
        params['subsample'] = trial.suggest_float("subsample", 0.7, 1.0)
        params['colsample_bytree'] = trial.suggest_float("colsample_bytree", 0.7, 1.0)
        params['max_depth'] = trial.suggest_int("max_depth", 2, 6)
        params['reg_lambda'] = trial.suggest_float("reg_lambda", .1, 50, log=True)
        params['learning_rate'] = trial.suggest_float("learning_rate", .005, 0.1, log=False)
        params['min_split_gain'] = trial.suggest_float("min_split_gain", 1e-5, 100., log=True)

        return params


class ObjectiveMultiLGB(ObjectiveMulti):

    def set_params(self, trial, params, prefix):
        params['min_child_samples'] = trial.suggest_int(f"{prefix}_min_child_samples", 1, 100, log=True)
        params['subsample'] = trial.suggest_float(f"{prefix}_subsample", 0.7, 1.0)
        params['colsample_bytree'] = trial.suggest_float(f"{prefix}_colsample_bytree", 0.7, 1.0)
        params['max_depth'] = trial.suggest_int(f"{prefix}_max_depth", 2, 6)
        params['reg_lambda'] = trial.suggest_float(f"{prefix}_reg_lambda", .1, 50, log=True)
        params['learning_rate'] = trial.suggest_float(f"{prefix}_learning_rate", .005, 0.1, log=False)
        params['min_split_gain'] = trial.suggest_float(f"{prefix}_min_split_gain", 1e-5, 100., log=True)

        return params


class ObjectiveCRF(Objective):

    def set_params(self, trial, params):
        params['criterion'] = trial.suggest_categorical("criterion", ["mse", "het"])
        params['honest'] = trial.suggest_categorical("honest", [False, True])

        params['max_depth'] = trial.suggest_int("max_depth", 2, 12)
        params['min_samples_leaf'] = trial.suggest_int("min_samples_leaf", 2, 100, log=True)

        params['max_features'] = trial.suggest_float("max_features", 0.2, 0.8)
        params['max_samples'] = trial.suggest_float("max_samples", 0.1, 0.5)

        params['min_balancedness_tol'] = trial.suggest_float("min_balancedness_tol", 0.05, 0.45)

        return params


class ObjectiveDR(Objective):
    def set_params(self, trial, params):
        params['hidden_scale'] = trial.suggest_float("hidden_scale", .5, 2.)
        params['outcome_scale'] = trial.suggest_float("outcome_scale", .5, 2.)

        params['alpha'] = trial.suggest_float("alpha", .5, 1.5)
        params['beta'] = trial.suggest_float("beta", .5, 1.5)
        params['batch_size'] = 2 ** trial.suggest_int("batch_size", 5, 10)

        params['learning_rate'] = trial.suggest_float("learning_rate", 1e-4, 1e-3)

        return params


class ObjectiveDCN(Objective):
    def set_params(self, trial, params):
        params['share_scale'] = trial.suggest_float("share_scale", .5, 2.)
        params['base_scale'] = trial.suggest_float("base_scale", .5, 2.)

        params['prpsy_w'] = trial.suggest_float("prpsy_w", 0.5, 1)
        params['escvr1_w'] = trial.suggest_float("escvr1_w", 0.5, 1)
        params['escvr0_w'] = trial.suggest_float("escvr0_w", 0.5, 1)
        params['mu0hat_w'] = trial.suggest_float("mu0hat_w", 0.5, 1)
        params['mu1hat_w'] = trial.suggest_float("mu1hat_w", 0.5, 1)

        params['steps_per_epoch'] = trial.suggest_int("steps_per_epoch", 100, 300)

        params['lr'] = trial.suggest_float("lr", 1e-4, 1e-3)

        return params
