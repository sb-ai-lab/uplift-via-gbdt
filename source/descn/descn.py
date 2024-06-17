import os
import shutil
import sys
import uuid
from time import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from torch.utils.data import TensorDataset, DataLoader

from .models import ShareNetwork, PrpsyNetwork, Mu1Network, Mu0Network, TauNetwork, ESX
from .util import wasserstein_torch, mmd2_torch


class Config:
    lr = 0.001
    decay_rate = 0.95
    decay_step_size = 1
    l2 = 0.001
    model_name = "model_name"

    n_experiments = 1
    batch_size = 3000
    share_dim = 128
    base_dim = 64
    reweight_sample = 1
    val_rate = 0.01
    do_rate = 0.1
    normalization = "divide"
    epochs = 10
    log_step = 100
    pred_step = 1
    optim = 'Adam'

    BatchNorm1d = True
    # loss weights
    prpsy_w = 0.5
    escvr1_w = 0.5
    escvr0_w = 1

    h1_w = 0
    h0_w = 0
    # ***sub space's loss weights
    mu0hat_w = 0.5
    mu1hat_w = 1

    # CFR loss
    # wass,mmd
    imb_dist = 'wass'
    # if imb_dist_w <=0 mean no use imb_dist_loss
    imb_dist_w = 0.0

    def __init__(self, **kwargs):
        for key in kwargs:
            self.__dict__[key] = kwargs[key]


class EarlyStopper:
    def __init__(self, temp_folder, patience=15, min_delta=0, ):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.temp_folder = temp_folder
        os.makedirs(temp_folder, exist_ok=False)

    def save(self, model):
        torch.save(model.state_dict(), os.path.join(self.temp_folder, 'checkpoint.pth'))

    def early_stop(self, validation_loss, model):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            self.save(model)
            # torch.save(model.state_dict(), os.path.join(self.temp_folder, 'checkpoint.pth'))


        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    def load(self, model):

        model.load_state_dict(torch.load(os.path.join(self.temp_folder, 'checkpoint.pth')))
        return model

    def clear(self):

        shutil.rmtree(self.temp_folder)


class DESCNNet:

    def __init__(
            self,
            share_scale=2.,
            base_scale=.5,
            steps_per_epoch=150,
            data_loader_num_workers=4,
            device='cuda',
            es=10,
            cat_cols=None,
            cat_params=None,
            **kwargs
    ):

        self.temp_folder = str(uuid.uuid4()) + f'_{time()}'

        self.share_scale = share_scale
        self.base_scale = base_scale
        self.steps_per_epoch = steps_per_epoch

        self.kwargs = kwargs

        self.num_workers = data_loader_num_workers

        self.train_dataloader = None
        self.valid_dataloader = None
        self.device = device
        self.scaler = StandardScaler()
        self.es = es
        self.cat_cols = [] if cat_cols is None else cat_cols
        if cat_params is None:
            cat_params = {
                'min_frequency': 10,
                'max_categories': 100,
                'handle_unknown': 'infrequent_if_exist',
                'sparse_output': False
            }
        self.enc = OneHotEncoder(**cat_params)

    def create_model(self, x):

        x = self.preprocess(x)
        nrows, input_dim = x.shape
        print(f'Train shape: {x.shape}')

        device = self.device

        share_dim = max(int(self.share_scale * input_dim), 4)
        base_dim = max(int(self.base_scale * share_dim), 2)

        batch_size = max(32, int(nrows / self.steps_per_epoch))

        print(f'Dataset specific params: share_dim={share_dim}; base_dim={base_dim}; batch_size={batch_size}')

        cfg = Config(share_dim=share_dim, base_dim=base_dim, batch_size=batch_size, **self.kwargs)
        self.cfg = cfg

        shareNetwork = ShareNetwork(input_dim=input_dim, share_dim=share_dim, base_dim=base_dim, cfg=cfg, device=device)
        prpsy_network = PrpsyNetwork(base_dim, cfg=cfg)
        mu1_network = Mu1Network(base_dim, cfg=cfg)
        mu0_network = Mu0Network(base_dim, cfg=cfg)
        tau_network = TauNetwork(base_dim, cfg=cfg)

        model = ESX(prpsy_network, mu1_network, mu0_network, tau_network, shareNetwork, cfg, device)
        self.model = model.to(device)
        optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.l2)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optim, step_size=cfg.decay_step_size,
            gamma=cfg.decay_rate
        )

        return x, cfg, optim, lr_scheduler

    def preprocess(self, x):

        if len(self.cat_cols) > 0:
            not_cat = np.setdiff1d(np.arange(x.shape[1]), self.cat_cols)
            x_cat = x[:, self.cat_cols]
            x = x[:, not_cat]

            try:
                x_cat = self.enc.transform(x_cat)
            except NotFittedError:
                x_cat = self.enc.fit_transform(x_cat)

        try:
            x = self.scaler.transform(x)
        except NotFittedError:
            x = self.scaler.fit_transform(x)

        if len(self.cat_cols) > 0:
            x = np.concatenate([x, x_cat], axis=1)

        return x

    def create_dataloaders(self, x, y, t, x_v=None, y_v=None, t_v=None):
        """
        Utility function to create train and validation data loader:

        Parameters
        ----------
        x: np.array
            covariates
        y: np.array
            target variable
        t: np.array
            treatment
        """

        x = torch.Tensor(x)
        t = torch.Tensor(t).reshape(-1, 1)
        y = torch.Tensor(y).reshape(-1, 1)
        train_dataset = TensorDataset(x, t, y)
        self.train_dataloader = DataLoader(
            train_dataset, batch_size=self.cfg.batch_size, num_workers=self.num_workers, shuffle=True, drop_last=True
        )

        if x_v is not None:
            x_v = self.preprocess(x_v)
            x_v = torch.Tensor(x_v)
            y_v = torch.Tensor(y_v).reshape(-1, 1)
            t_v = torch.Tensor(t_v).reshape(-1, 1)
            valid_dataset = TensorDataset(x_v, t_v, y_v)
            self.valid_dataloader = DataLoader(
                valid_dataset, batch_size=self.cfg.batch_size, num_workers=self.num_workers, shuffle=False
            )

    def fit(self, x, y, t, x_v=None, y_v=None, t_v=None):
        """
        Function used to train the dragonnet model

        Parameters
        ----------
        x: np.array
            covariates
        y: np.array
            target variable
        t: np.array
            treatment
        """
        x, cfg, optim, lr_scheduler = self.create_model(x)

        self.create_dataloaders(x, y, t, x_v, y_v, t_v)
        early_stopper = EarlyStopper(self.temp_folder, patience=self.es, min_delta=0)
        early_stopper.save(self.model)

        for epoch in range(self.cfg.epochs):

            self.model.train()

            for batch, (X, tr, y1) in enumerate(self.train_dataloader):
                X, tr, y1 = X.to(self.device), tr.to(self.device), y1.to(self.device)

                p_prpsy_logit, p_estr, p_escr, p_tau_logit, p_mu1_logit, p_mu0_logit, p_prpsy, p_mu1, p_mu0, p_h1, p_h0, shared_h \
                    = self.model(X)

                # y0_pred, y1_pred, t_pred, eps = self.model(X)
                try:

                    loss = self.loss_f(
                        tr, y1, p_prpsy_logit, p_estr, p_escr, p_tau_logit,
                        p_mu1_logit, p_mu0_logit, p_prpsy, p_mu1, p_mu0, p_h1, p_h0, shared_h
                    )
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                except Exception:
                    continue

            lr_scheduler.step()

            if self.valid_dataloader:
                valid_loss = self.validate_step()
                print(
                    f"epoch: {epoch}--------- train_loss: {loss} ----- valid_loss: {valid_loss}"
                )
                if early_stopper.early_stop(valid_loss, self.model):
                    break
            else:
                print(f"epoch: {epoch}--------- train_loss: {loss}")

        self.model = early_stopper.load(self.model)
        early_stopper.clear()

    def loss_f(self, t_labels, y_labels, *args):

        p_prpsy_logit, p_estr, p_escr, p_tau_logit, p_mu1_logit, p_mu0_logit, p_prpsy, p_mu1, p_mu0, p_h1, p_h0, shared_h = args

        e_labels = torch.zeros_like(t_labels).to(t_labels.device)

        p_t = torch.mean(t_labels).item()
        if self.cfg.reweight_sample:
            w_t = t_labels / (
                    2 * p_t)
            w_c = (1 - t_labels) / (2 * (1 - p_t))
            sample_weight = w_t + w_c
        else:
            sample_weight = torch.ones_like(t_labels)
            p_t = 0.5

        # set loss functions
        sample_weight = sample_weight[~e_labels.bool()]
        loss_w_fn = nn.BCELoss(weight=sample_weight)
        loss_fn = nn.BCELoss()
        loss_mse = nn.MSELoss()
        loss_with_logit_fn = nn.BCEWithLogitsLoss()  # for logit
        loss_w_with_logit_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(1 / (2 * p_t)))  # for propensity loss

        # calc loss
        # p_prpsy_logit, p_estr, p_escr, p_tau_logit, p_mu1_logit, p_mu0_logit, p_prpsy, p_mu1, p_mu0, p_h1, p_h0, shared_h

        # try:
        # loss for propensity
        prpsy_loss = self.cfg.prpsy_w * loss_w_with_logit_fn(p_prpsy_logit[~e_labels.bool()],
                                                             t_labels[~e_labels.bool()])
        # loss for ESTR, ESCR
        estr_loss = self.cfg.escvr1_w * loss_w_fn(p_estr[~e_labels.bool()],
                                                  (y_labels * t_labels)[~e_labels.bool()])
        escr_loss = self.cfg.escvr0_w * loss_w_fn(p_escr[~e_labels.bool()],
                                                  (y_labels * (1 - t_labels))[~e_labels.bool()])

        # loss for TR, CR
        tr_loss = self.cfg.h1_w * loss_fn(p_h1[t_labels.bool()],
                                          y_labels[t_labels.bool()])  # * (1 / (2 * p_t))
        cr_loss = self.cfg.h0_w * loss_fn(p_h0[~t_labels.bool()],
                                          y_labels[~t_labels.bool()])  # * (1 / (2 * (1 - p_t)))

        # loss for cross TR: mu1_prime, cross CR: mu0_prime
        cross_tr_loss = self.cfg.mu1hat_w * loss_fn(torch.sigmoid(p_mu0_logit + p_tau_logit)[t_labels.bool()],
                                                    y_labels[t_labels.bool()])
        cross_cr_loss = self.cfg.mu0hat_w * loss_fn(torch.sigmoid(p_mu1_logit - p_tau_logit)[~t_labels.bool()],
                                                    y_labels[~t_labels.bool()])

        imb_dist = 0
        if self.cfg.imb_dist_w > 0:
            if self.cfg.imb_dist == "wass":
                imb_dist = wasserstein_torch(X=shared_h, t=t_labels)
            elif self.cfg.imb_dist == "mmd":
                imb_dist = mmd2_torch(shared_h, t_labels)
            else:
                sys.exit(1)
        imb_dist_loss = self.cfg.imb_dist_w * imb_dist

        total_loss = prpsy_loss + estr_loss + escr_loss \
                     + tr_loss + cr_loss \
                     + cross_tr_loss + cross_cr_loss \
                     + imb_dist_loss

        return total_loss

    def validate_step(self):
        """
        Calculates validation loss

        Returns
        -------
        valid_loss: torch.Tensor
            validation loss
        """

        self.model.eval()

        valid_loss = []
        with torch.no_grad():
            for batch, (X, tr, y1) in enumerate(self.valid_dataloader):

                X, tr, y1 = X.to(self.device), tr.to(self.device), y1.to(self.device)
                p_prpsy_logit, p_estr, p_escr, p_tau_logit, p_mu1_logit, p_mu0_logit, p_prpsy, p_mu1, p_mu0, p_h1, p_h0, shared_h \
                    = self.model(X)

                try:
                    loss = self.loss_f(
                        tr, y1, p_prpsy_logit, p_estr, p_escr, p_tau_logit,
                        p_mu1_logit, p_mu0_logit, p_prpsy, p_mu1, p_mu0, p_h1, p_h0, shared_h
                    )
                except Exception:
                    continue

                valid_loss.append(loss)
        return torch.Tensor(valid_loss).mean()

    def predict(self, x):
        """
        Function used to predict on covariates.

        Parameters
        ----------
        x: torch.Tensor or numpy.array
            covariates

        Returns
        -------

        """
        self.model.eval()

        res = np.zeros((x.shape[0],), dtype=np.float32)
        x = self.preprocess(x)
        x = torch.Tensor(x)

        ds = TensorDataset(x)
        dl = DataLoader(
            ds, batch_size=self.cfg.batch_size, num_workers=self.num_workers, shuffle=False
        )

        with torch.no_grad():
            for n, (batch,) in enumerate(dl):
                batch = batch.to(self.device)
                p_prpsy_logit, p_estr, p_escr, p_tau_logit, p_mu1_logit, p_mu0_logit, p_prpsy, p_mu1, p_mu0, p_h1, p_h0, shared_h \
                    = self.model(batch)
                res[n * self.cfg.batch_size: (n + 1) * self.cfg.batch_size] = (p_h1 - p_h0).detach().cpu().numpy()[:, 0]

        return res
