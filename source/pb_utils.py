import cupy as cp
import numpy as np
from py_boost.callbacks.callback import Callback
from py_boost.gpu.boosting import GradientBoosting
from py_boost.gpu.losses import Loss, BCELoss, BCEMetric, MSELoss, Metric
from py_boost.multioutput.sketching import GradSketch, RandomProjectionSketch, RandomSamplingSketch
from py_boost.multioutput.target_splitter import SingleSplitter


def get_treshold_stats(fact, pred):
    order = pred.argsort()[::-1]

    sorted_y = fact[order]
    uplift = pred[order]

    idx = cp.r_[
        cp.nonzero(cp.diff(uplift))[0],
        order.shape[0] - 1
    ]
    cs = cp.nancumsum(sorted_y, axis=0)[idx]
    cc = cp.nancumsum((~cp.isnan(sorted_y)).astype(cp.float32), axis=0)[idx]

    return cs, cc, idx


def get_qini_curve(fact, pred):
    cs, cc, idx = get_treshold_stats(fact, pred)
    curve = cs[:, 1] - cp.where(cc[:, 0] > 0, cs[:, 0] * cc[:, 1] / cc[:, 0], 0)
    return idx + 1, curve


def get_perfect_and_baseline_qini(fact):
    y_ = cp.nansum(fact, axis=1)
    t_ = cp.isnan(fact[:, 0]).astype(cp.float32)

    perfect = (y_ * t_ - y_ * (1 - t_))

    x, y = get_qini_curve(fact, perfect)
    x = cp.r_[0, x]
    y = cp.r_[0, y]
    score_perfect = float(cp.trapz(y, x))

    x, y = np.array([0, float(x[-1])]), np.array([0, float(y[-1])])
    score_baseline = np.trapz(y, x)

    return score_perfect, score_baseline


class BCEwithNaNMetric(BCEMetric):

    def __call__(self, y_true, y_pred, sample_weight=None):
        mask = ~cp.isnan(y_true)

        err = super().error(cp.where(mask, y_true, 0), y_pred)
        err = err * mask

        if sample_weight is not None:
            err = err * sample_weight
            mask = mask * sample_weight

        return float(err.sum() / mask.sum())


class BCEWithNaNLoss(BCELoss):

    def __init__(self, uplift=False):
        self.uplift = uplift
        self.clip_value = 1e-6

    def base_score(self, y_true):
        # Replace .mean with nanmean function to calc base score
        means = cp.nanmean(y_true, axis=0)
        means = cp.where(cp.isnan(means), 0, means)
        means = cp.clip(means, self.clip_value, 1 - self.clip_value)

        return cp.log(means / (1 - means))
        # return cp.zeros(y_true.shape[1], dtype=cp.float32)

    def get_grad_hess(self, y_true, y_pred):
        # first, get nan mask for y_true
        mask = cp.isnan(y_true)
        # then, compute loss with any values at nan places just to prevent the exception
        grad, hess = super().get_grad_hess(cp.where(mask, 0, y_true), y_pred)
        # invert mask
        mask = (~mask).astype(cp.float32)
        # multiply grad and hess on inverted mask
        # now grad and hess eq. 0 on NaN points
        # that actually means that prediction on that place should not be updated
        grad = grad * mask
        hess = hess * mask

        return grad, hess

    def postprocess_output(self, y_pred):
        y_pred = super().postprocess_output(y_pred)

        if self.uplift:
            uplift = y_pred[:, 1:] - y_pred[:, :1]

            return uplift

        return y_pred


class MSEWithNaNLoss(MSELoss):

    def get_grad_hess(self, y_true, y_pred):
        # first, get nan mask for y_true
        mask = cp.isnan(y_true)
        # then, compute loss with any values at nan places just to prevent the exception
        grad, hess = super().get_grad_hess(cp.where(mask, 0, y_true), y_pred)
        # invert mask
        mask = (~mask).astype(cp.float32)
        # multiply grad and hess on inverted mask
        # now grad and hess eq. 0 on NaN points
        # that actually means that prediction on that place should not be updated
        grad = grad * mask
        hess = hess * mask

        return grad, hess


class QINIMetric(Metric, Callback):

    def __init__(self, freq=1):

        self.freq = freq
        self.value = None
        self.n = None
        self.base = None
        self.perf = None
        self.trt_sl = None

        self.last_score = None

    def before_iteration(self, build_info):

        self.n = build_info['num_iter']

    def before_train(self, build_info):

        y_true = build_info['data']['valid']['target']
        assert len(y_true) <= 1, 'Only single dataset is avaliable to evaluate'
        y_true = y_true[0]

        nnans = ~np.isnan(y_true)
        self.trt_sl = nnans[:, 1:] | nnans[:, :1]

        self.n = None
        self.base, self.perf = [], []

        for i in range(y_true.shape[1] - 1):
            cols = [0, i + 1]
            sl = cp.nonzero(self.trt_sl[:, i])[0]
            fact = y_true[:, cols][sl]
            perf, base = get_perfect_and_baseline_qini(fact)
            self.perf.append(perf)
            self.base.append(base)

        return

    def after_train(self, build_info):

        self.__init__(self.freq)

    def __call__(self, y_true, y_pred, sample_weight=None):

        if (self.n % self.freq) == 0:

            qinis = []

            for i in range(y_pred.shape[1]):
                cols = [0, i + 1]
                sl = cp.nonzero(self.trt_sl[:, i])[0]
                fact = y_true[:, cols][sl]
                pred = y_pred[:, i][sl]

                x, y = get_qini_curve(fact, pred)
                q = float(cp.trapz(y, x))
                score = (q - self.base[i]) / (self.perf[i] - self.base[i])
                qinis.append(score)

            self.last_score = np.mean(qinis)

        return self.last_score

    def compare(self, v0, v1):

        return v0 > v1


class UpliftSketch(GradSketch):

    def __call__(self, grad, hess):
        grad = grad.sum(axis=1, keepdims=True)
        hess = hess.sum(axis=1, keepdims=True)
        # hess = cp.ones_like(grad)

        return grad, hess


class MixedUpliftSketch(UpliftSketch):

    def __init__(self):
        self.base_sketch = RandomProjectionSketch(1)

    def __call__(self, grad, hess):
        bg, bh = self.base_sketch(grad, hess)
        ug, uh = super().__call__(grad, hess)

        return cp.concatenate([bg, ug], axis=1), uh


class RFCallback(Callback):

    def process(self, ens, last_pred, base_score, n, lr):

        # clean ensemble from prediction
        ens = ens - last_pred - base_score
        # add as mean
        n = n + 1
        ens = base_score + ens * ((n - 1) / n) + last_pred / (n * lr)

        return ens

    def after_iteration(self, build_info):

        train = build_info['data']['train']
        valid = build_info['data']['valid']

        train['ensemble'][:] = build_info['model'].base_score

        for i in range(len(valid['ensemble'])):
            valid['ensemble'][i] = self.process(
                valid['ensemble'][i],
                valid['last_tree']['preds'][i],
                build_info['model'].base_score,
                build_info['num_iter'],
                build_info['model'].lr
            )

        return False

    def after_train(self, build_info):

        model = build_info['model']
        trees, lr = model.models, model.lr
        n = len(trees)

        for i in range(n):
            trees[i].values = trees[i].values / (lr * n)

        return


class RandomSamplingSketchX(RandomSamplingSketch):

    def before_iteration(self, build_info):
        super().before_iteration(build_info)
        self.num_iter = build_info['num_iter']

    def __call__(self, grad, hess):
        if np.random.rand() > .8:
            return grad, hess

        return super().__call__(grad, hess)


class PyBoostClassifier:

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.model = None
        self.loss = 'bce'

    def fit(self, X, y, sample_weight=None):
        print(X.shape, y.shape)

        self.model = GradientBoosting(self.loss, *self.args, **self.kwargs)
        self.model.fit(X, y, sample_weight=sample_weight)

        return self

    def predict_proba(self, X):
        pred = self.model.predict(X)
        return np.concatenate([1 - pred, pred], axis=1)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def fit_predict(self, X, y, sample_weight=None):
        self.fit(X, y, sample_weight=sample_weight)
        return self.predict(X)


class PyBoostRegressor(PyBoostClassifier):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = 'mse'

    def predict(self, X):
        return self.model.predict(X)[:, 0]


class ComposedUpliftLoss(Loss, Callback):

    def __init__(self, base_loss, start_iter=10, weight=.5, masked=False):

        self.base_loss = base_loss
        self.uplift_loss = MSEWithNaNLoss()
        self.start_iter = start_iter
        self.n = 0
        self.weight = weight
        self.masked = masked

    def before_iteration(self, build_info):

        self.n = build_info['num_iter']

    def base_score(self, y_true):

        score_base = self.base_loss.base_score(y_true)
        score_upl = cp.nanmean(y_true, axis=0)
        score_upl = score_upl[1:] - score_upl[:1]

        return cp.concatenate([score_base, score_upl])

    def get_grad_hess(self, y_true, y_pred):

        l = y_true.shape[1]

        grad, hess = self.base_loss.get_grad_hess(y_true, y_pred[:, :l])
        #         if self.n < self.start_iter:
        #         not_null_mask = ~np.isnan(y_true)
        #         proxy = cp.where(not_null_mask, y_true, self.base_loss.postprocess_output(y_pred[:, :l]))
        #         uplift = proxy[:, 1:] - proxy[:, :1]
        #         if self.masked:
        #             mask = not_null_mask[:, 1:] | not_null_mask[:, :1] # mask the cells with al least one value real
        #             uplift = cp.where(mask, uplift, np.nan)

        proxy = self.base_loss.postprocess_output(
            y_pred[:, :l])  # cp.where(not_null_mask, y_true, self.base_loss.postprocess_output(y_pred[:, :l]))
        uplift = proxy[:, 1:] - proxy[:, :1]
        #         if self.masked:
        #             mask = not_null_mask[:, 1:] | not_null_mask[:, :1] # mask the cells with al least one value real
        #             uplift = cp.where(mask, uplift, np.nan)

        if self.n >= self.start_iter:
            # print('Good branch')
            grad_upl, _ = self.uplift_loss(uplift, y_pred[:, l:])
            # print(grad_upl.mean(axis=0), grad_upl.std(axis=0))
        else:
            grad_upl = cp.zeros((grad.shape[0], grad.shape[1] - 1), dtype=cp.float32)

        hess_upl = cp.ones_like(grad_upl)
        #         if self.masked:
        #             hess_upl = mask.astype(cp.float32)
        #         else:
        #             hess_upl = cp.ones_like(grad_upl)

        grad = cp.concatenate([grad, grad_upl], axis=1)
        hess = cp.concatenate([hess, hess_upl], axis=1)

        return grad, hess

    def postprocess_output(self, y_pred):

        # print(y_pred.shape, y_pred.mean(axis=0), y_pred.std(axis=0))

        l = y_pred.shape[1] // 2 + 1

        base_pred = self.base_loss.postprocess_output(y_pred[:, :l])
        base_pred = base_pred[:, 1:] - base_pred[:, :1]
        y_pred = y_pred[:, l:]

        # print(y_pred.std(axis=0), base_pred.std(axis=0))

        return y_pred * self.weight + base_pred * (1 - self.weight)


class UpliftSplitter(SingleSplitter):

    def before_iteration(self, build_info):
        """Initialize indexers

        Args:
            build_info: dict

        Returns:

        """
        if build_info['num_iter'] == 0:
            nout = build_info['data']['train']['grad'].shape[1] // 2 + 1

            self.indexer = [cp.arange(nout, dtype=cp.uint64), cp.arange(nout, nout * 2 - 1, dtype=cp.uint64)]

    def __call__(self):
        """Get list of indexers for each group

        Returns:
            list of cp.ndarrays of indexers
        """
        return self.indexer


class UpliftSplitterXN(SingleSplitter):

    def before_iteration(self, build_info):
        """Initialize indexers

        Args:
            build_info: dict

        Returns:

        """
        if build_info['num_iter'] == 0:
            nout = build_info['data']['train']['grad'].shape[1] // 2 + 1

            self.indexer = [
                               cp.arange(nout, dtype=cp.uint64),
                               # cp.arange(nout, nout * 2 - 1, dtype=cp.uint64)
                           ] + [cp.asarray([x], dtype=cp.uint64) for x in range(nout, nout * 2 - 1)]

    def __call__(self):
        """Get list of indexers for each group

        Returns:
            list of cp.ndarrays of indexers
        """
        return self.indexer


class ComposedUpliftSketch(GradSketch):

    def __init__(self, base_sketch=None):

        self.base_sketch = base_sketch
        if base_sketch is None:
            self.base_sketch = GradSketch()

        self.flg = False

    def before_iteration(self, build_info):

        self.flg = False

    def __call__(self, grad, hess):

        if self.flg:
            return self.base_sketch(grad, hess)

        self.flg = True
        grad = grad.sum(axis=1, keepdims=True)
        hess = hess.sum(axis=1, keepdims=True)
        return grad, hess


class ComposedUpliftSketchMix(GradSketch):

    def __init__(self, base_sketch=None):

        self.base_sketch = base_sketch
        if base_sketch is None:
            self.base_sketch = GradSketch()

        self.upl_sketch = MixedUpliftSketch()

        self.flg = False

    def before_iteration(self, build_info):

        self.flg = False

    def __call__(self, grad, hess):

        if self.flg:
            return self.base_sketch(grad, hess)

        self.flg = True
        grad, hess = self.upl_sketch(grad, hess)
        return grad, hess
