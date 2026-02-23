import pandas as pd
import numpy as np
import os

def prepare_npz(args):
    # extract args
    L = args.n_look_back
    # load csv
    path = os.path.join(args.data_dir, f'{args.dataset}.csv')
    df = pd.read_csv(path, header=None)
    # extract raw time series
    X = df.values.reshape(-1, 15, 15)[:, 5:10, 5:10]
    T = X.shape[0]
    # split train test
    T_train = round(T * 0.8)
    # export train data
    train_data = {
        'X': X[:T_train],
        'Y': X[L:T_train+L],
    }
    path = os.path.join(args.data_dir, f'{args.dataset}_train.npz')
    np.savez_compressed(path, **train_data)
    # export test data
    test_data = {
        'X': X[T_train:-L],
        'Y': X[T_train+L:],
    }
    path = os.path.join(args.data_dir, f'{args.dataset}_test.npz')
    np.savez_compressed(path, **test_data)
