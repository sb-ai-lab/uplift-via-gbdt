from copy import deepcopy

from causalml.inference.meta import BaseDRRegressor
from causalml.inference.meta import BaseTClassifier, BaseXClassifier, BaseRClassifier
from econml.grf import CausalForest
from py_boost import GradientBoosting
from xgboost import XGBRegressor, XGBClassifier

from .descn import DESCNNet
from .dragonnet import DragonNet
from .pb_utils import ComposedUpliftLoss, QINIMetric, BCEWithNaNLoss, UpliftSplitter, UpliftSplitterXN, \
    RandomSamplingSketchX


def get_tlearner_xgb(params, estimator=XGBClassifier):
    model = BaseTClassifier(
        control_learner=estimator(**params['control_learner']),
        treatment_learner=estimator(**params['treatment_learner']),
        control_name='control'
    )

    return model


def get_xlearner_xgb(params, estimator=XGBClassifier, regressor=XGBRegressor):
    model = BaseXClassifier(
        control_outcome_learner=estimator(**params['control_outcome_learner']),
        treatment_outcome_learner=estimator(**params['treatment_outcome_learner']),
        control_effect_learner=regressor(**params['control_effect_learner']),
        treatment_effect_learner=regressor(**params['treatment_effect_learner']),
        control_name='control'
    )

    return model


def get_rlearner_xgb(params, estimator=XGBClassifier, regressor=XGBRegressor):
    model = BaseRClassifier(
        outcome_learner=estimator(**params['outcome_learner']),
        effect_learner=regressor(**params['effect_learner']),
        control_name='control'
    )

    return model


class XGBRegClassifier(XGBClassifier):

    def predict(self, X):
        return self.predict_proba(X)[:, 1]


def get_drlearner_xgb(params, estimator=XGBRegClassifier, regressor=XGBRegressor):
    model = BaseDRRegressor(
        control_outcome_learner=estimator(**params['control_outcome_learner']),
        treatment_outcome_learner=estimator(**params['treatment_outcome_learner']),
        treatment_effect_learner=regressor(**params['treatment_effect_learner']),

        control_name='control'
    )

    return model


def get_pb_uplift(params, xn=True, masked=True, weight=.5):
    params = deepcopy(params['params'])

    loss = ComposedUpliftLoss(BCEWithNaNLoss(), 1, weight=weight, masked=masked)
    metric = QINIMetric(freq=10)
    splitter = UpliftSplitterXN() if xn else UpliftSplitter()

    model = GradientBoosting(
        loss, metric,
        target_splitter=splitter,
        multioutput_sketch=RandomSamplingSketchX(1, smooth=1),
        callbacks=[
            loss,
            metric,
        ],
        **params
    )

    return model


def get_crf(params, ):
    model = CausalForest(
        **params['params']
    )

    return model


def get_drnet(params, cat_cols=None):
    model = DragonNet(
        cat_cols=cat_cols, **params['params']
    )

    return model


def get_dcn(params, cat_cols=None):
    model = DESCNNet(
        cat_cols=cat_cols, **params['params']
    )

    return model
