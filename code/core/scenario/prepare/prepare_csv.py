import pandas as pd
import pickle
import os

def prepare_csv(args):
    path = os.path.join(args.data_dir, f'{args.dataset}.pkl')
    with open(path, 'rb') as fp:
        df = pickle.load(fp)
    path = os.path.join(args.data_dir, f'{args.dataset}.csv')
    df.to_csv(path, index=None, header=None)
