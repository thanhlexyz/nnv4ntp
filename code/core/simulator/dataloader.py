from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import os

class TrafficPredictionDataset(Dataset):

    def __init__(self, mode, args):
        # save args
        self.args = args
        # load data
        path = os.path.join(args.data_dir, f'{args.dataset}_{mode}.npz')
        data = np.load(path)
        # save data
        self.X = torch.tensor(data['X'], dtype=torch.float32)
        self.Y = torch.tensor(data['Y'][:, 3, 3], dtype=torch.float32)

    def __len__(self):
        L = self.args.n_look_back
        return len(self.X) - L

    def __getitem__(self, idx):
        L = self.args.n_look_back
        x, y = self.X[idx:idx+L].to(self.args.device), self.Y[idx].to(self.args.device)
        x = x
        y = y[None]
        return x, y

def create(args):
    # load dataset
    train_dataset = TrafficPredictionDataset('train', args)
    test_dataset = TrafficPredictionDataset('test', args)
    # create loader
    # if args.mode == 'test':
    #     args.batch_size = 1
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    return train_loader, test_loader
