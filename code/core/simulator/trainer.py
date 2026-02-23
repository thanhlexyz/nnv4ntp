import torch.optim as optim
import torch.nn as nn
import numpy as np
import simulator
import torch
import time
import os


import torch
import torch.nn as nn

class CapacityProvisioningLoss(nn.Module):

    def __init__(self, alpha, epsilon):
        super().__init__()
        self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Broadcast alpha if scalar
        alpha = self.alpha
        epsilon = self.epsilon

        diff = y_pred - y_true

        y1 = - epsilon * diff + alpha
        y2 = - (1.0 / epsilon) * diff + alpha

        denom = torch.tensor(1.0 - (epsilon * alpha))
        # Guard against division by zero (or near-zero)
        denom_safe = torch.where(torch.abs(denom) < 1e-8,
                                 torch.full_like(denom, 1e-8),
                                 denom)
        y3 = (alpha / denom_safe) * (diff - (epsilon * alpha))

        cost = torch.zeros_like(diff)
        cost = torch.where(diff > (epsilon * alpha), y3, cost)
        cost = torch.where(diff < 0, y1, cost)
        mid_mask = (diff <= (epsilon * alpha)) & (diff >= 0)
        cost = torch.where(mid_mask, y2, cost)
        return cost.mean()

class Trainer:

    def __init__(self, args):
        # save args
        self.args = args
        # create dataloader
        self.train_loader, self.test_loader = simulator.dataloader.create(args)
        # create model
        self.model = simulator.model.create(args)
        # create monitor
        n_monitor_step = args.n_train_epoch if args.mode == 'train' else len(self.test_loader.dataset)
        self.monitor = simulator.monitor.create(n_monitor_step, args)
        # other
        self.step = 0
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self.criterion = CapacityProvisioningLoss(alpha=args.deepcog_alpha, epsilon=args.deepcog_epsilon) # nn.MSELoss()

    def train_epoch(self):
        # extract args
        criterion = self.criterion
        args = self.args
        # initialize
        losses = []
        for (x, y) in self.train_loader:
            # extract data and model
            model     = self.model
            optimizer = self.optimizer
            # inference
            y_hat = model(x)
            # train
            loss = criterion(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # report
            losses.append(loss.item())
            # count
            self.step += 1
        def mse_fn(y, y_hat):
            return torch.mean((y - y_hat) ** 2)
        def over_percent_fn(y, y_hat):
            return torch.mean(torch.relu(y_hat - y) / y)
        def under_percent_fn(y, y_hat):
            return torch.mean(torch.relu(y - y_hat) / y)
        # result
        info = {
            'loss': np.mean(losses),
            'step': self.step,
            'mse' : mse_fn(y, y_hat).item(),
            'over_percent' : over_percent_fn(y, y_hat).item(),
            'under_percent' : under_percent_fn(y, y_hat).item(),
        }
        return info

    def train(self):
        # extract args
        args = self.args
        # train
        for epoch in range(args.n_train_epoch):
            info = {'epoch': epoch}
            info.update(self.train_epoch())
            self.monitor.step(info)
            self.monitor.export_csv()
            if not epoch % args.n_save_epoch:
                self.save()

    def test_epoch(self):
        # extract args
        args = self.args
        # BEGIN: define metric functions
        def mse_fn(y, y_hat):
            return torch.mean((y - y_hat) ** 2)
        def rmse_fn(y, y_hat):
            return torch.sqrt(mse_fn(y, y_hat))
        def mae_fn(y, y_hat):
            return torch.mean(torch.abs(y - y_hat))
        def over_percent_fn(y, y_hat):
            return torch.mean(torch.relu(y_hat - y) / y)
        def under_percent_fn(y, y_hat):
            return torch.mean(torch.relu(y - y_hat) / y)
        # END
        for (x, y) in self.test_loader:
            # extract data and model for bs l
            model = self.model
            # inference
            y_hat = model(x)
            # count
            self.step += 1
            # report result
            info = {
                'step': self.step,
                'mse' : mse_fn(y, y_hat).item(),
                'mae' : mae_fn(y, y_hat).item(),
                'rmse': rmse_fn(y, y_hat).item(),
                'over_percent' : over_percent_fn(y, y_hat).item(),
                'under_percent' : under_percent_fn(y, y_hat).item(),
            }
            self.monitor.step(info)
            self.monitor.export_csv()

    def test(self):
        # extract args
        args = self.args
        # train
        for epoch in range(args.n_test_epoch):
            self.test_epoch()
            if not epoch % args.n_save_epoch:
                self.save()

    def save(self):
        args = self.args
        path = os.path.join(args.model_dir, f'{args.dataset}.pth')
        torch.save(self.model.state_dict(), path)

    def load(self):
        args = self.args
        path = os.path.join(args.model_dir, f'{args.dataset}.pth')
        if os.path.exists(path):
            state_dict = torch.load(path)
            self.model.load_state_dict(state_dict)
            print(f'[+] loaded {path}')

def create(args):
    return Trainer(args)
