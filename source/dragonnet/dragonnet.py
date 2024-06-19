import uuid
from functools import partial

import numpy as np
import torch
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from torch.utils.data import TensorDataset, DataLoader

from .model import DragonNetBase, dragonnet_loss, tarreg_loss, EarlyStopper


class DragonNet:
    """
    Main class for the Dragonnet model

    Parameters
    ----------
    input_dim: int
        input dimension for convariates
    shared_hidden: int, default=200
        layer size for hidden shared representation layers
    outcome_hidden: int, default=100
        layer size for conditional outcome layers
    alpha: float, default=1.0
        loss component weighting hyperparameter between 0 and 1
    beta: float, default=1.0
        targeted regularization hyperparameter between 0 and 1
    epochs: int, default=200
        Number training epochs
    steps_per_epoch: int, default=100
        Number of steps per epoch to scale batch size
    learning_rate: float, default=1e-3
        Learning rate
    data_loader_num_workers: int, default=4
        Number of workers for data loader
    loss_type: str, {'tarreg', 'default'}, default='tarreg'
        Loss function to use
    """

    def __init__(
            self,
            hidden_scale=2.,
            outcome_scale=.5,
            alpha=1.0,
            beta=1.0,
            epochs=200,
            steps_per_epoch=100,
            learning_rate=1e-5,
            data_loader_num_workers=4,
            loss_type="tarreg",
            device='cuda',
            es=10,
            cat_cols=None,
            cat_params=None
    ):

        self.temp_folder = str(uuid.uuid1())

        self.hidden_scale = hidden_scale
        self.outcome_scale = outcome_scale
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = None
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

        if loss_type == "tarreg":
            self.loss_f = partial(tarreg_loss, alpha=alpha, beta=beta)
        elif loss_type == "default":
            self.loss_f = partial(dragonnet_loss, alpha=alpha)

    def create_model(self, x):

        x = self.preprocess(x)
        nrows, input_dim = x.shape

        self.batch_size = max(32, int(nrows / self.steps_per_epoch))

        shared_hidden = max(int(self.hidden_scale * input_dim), 4)
        outcome_hidden = max(int(self.outcome_scale * shared_hidden), 2)
        self.model = DragonNetBase(input_dim, shared_hidden, outcome_hidden).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        return x

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
            train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True
        )

        if x_v is not None:
            x_v = self.preprocess(x_v)
            x_v = torch.Tensor(x_v)
            y_v = torch.Tensor(y_v).reshape(-1, 1)
            t_v = torch.Tensor(t_v).reshape(-1, 1)
            valid_dataset = TensorDataset(x_v, t_v, y_v)
            self.valid_dataloader = DataLoader(
                valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False
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
        valid_perc: float
            Percentage of data to allocate to validation set
        """
        x = self.create_model(x)
        self.create_dataloaders(x, y, t, x_v, y_v, t_v)
        early_stopper = EarlyStopper(self.temp_folder, patience=self.es, min_delta=0)
        for epoch in range(self.epochs):

            self.model.train()

            for batch, (X, tr, y1) in enumerate(self.train_dataloader):
                X, tr, y1 = X.to(self.device), tr.to(self.device), y1.to(self.device)
                y0_pred, y1_pred, t_pred, eps = self.model(X)
                loss = self.loss_f(y1, tr, t_pred, y0_pred, y1_pred, eps)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
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
                y0_pred, y1_pred, t_pred, eps = self.model(X)
                loss = self.loss_f(y1, tr, t_pred, y0_pred, y1_pred, eps)
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
        y0_pred: torch.Tensor
            outcome under control
        y1_pred: torch.Tensor
            outcome under treatment
        t_pred: torch.Tensor
            predicted treatment
        eps: torch.Tensor
            trainable epsilon parameter
        """
        self.model.eval()

        res = np.zeros((x.shape[0],), dtype=np.float32)
        x = self.preprocess(x)
        x = torch.Tensor(x)

        ds = TensorDataset(x)
        dl = DataLoader(
            ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False
        )

        with torch.no_grad():
            for n, (batch,) in enumerate(dl):
                batch = batch.to(self.device)
                y0_pred, y1_pred, t_pred, eps = self.model(batch)
                res[n * self.batch_size: (n + 1) * self.batch_size] = (y1_pred - y0_pred).detach().cpu().numpy()[:, 0]

        return res
