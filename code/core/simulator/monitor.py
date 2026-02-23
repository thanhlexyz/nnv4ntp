import pandas as pd
import numpy as np
import simulator
import torch
import tqdm
import os

class Monitor:

    def __init__(self, n_step, args):
        # save
        self.args = args
        # initialize progress bar
        if not args.no_pbar:
            self.bar = tqdm.tqdm(range(n_step))
        # initialize writer
        self.csv_data = {}
        self.global_step = 0

    def __update_time(self):
        if not self.args.no_pbar:
            self.bar.update(1)

    def __update_description(self, **kwargs):
        _kwargs = {}
        for key in kwargs:
            for name in ['loss', 'mse', 'percent']:
                if name in key:
                    _kwargs[key] = f'{kwargs[key]:0.3f}'
                elif 'step' in key:
                    _kwargs[key] = f'{kwargs[key]:d}'
        if not self.args.no_pbar:
            self.bar.set_postfix(**_kwargs)

    def __display(self):
        if not self.args.no_pbar:
            self.bar.display()
            print()

    def step(self, info):
        # extract stats from all stations
        # update progress bar
        self.__update_time()
        self.__update_description(**info)
        self.__display()
        # log to csv
        self.__update_csv(info)
        self.global_step += 1

    @property
    def label(self):
        return simulator.util.get_label(self.args)

    def __update_csv(self, info):
        for key in info.keys():
            if key not in self.csv_data:
                self.csv_data[key] = [float(info[key])]
            else:
                self.csv_data[key].append(float(info[key]))

    def export_csv(self):
        args = self.args
        path = os.path.join(args.csv_dir, f'{self.label}.csv')
        df = pd.DataFrame(self.csv_data)
        df.to_csv(path, index=None)

def create(n_step, args):
    return Monitor(n_step, args)
