import pandas as pd
import os

def table1(args):
    # setup
    etas = [0.01, 0.05, 0.1, 0.15, 0.2]
    zetas = [0.01, 0.05, 0.1]
    # load csv
    path = os.path.join(args.csv_dir, 'verify.csv')
    df = pd.read_csv(path)
    df = df[df['v_output_type'] == 'under']
    pres = [
        "$\\zeta=1\%$",
        "$\\zeta=5\%$",
        "$\\zeta=10\%$",
    ]
    posts = [
        " \\\\ ",
        " \\\\ ",
        " \\\\ \\bottomrule"
    ]
    for i, zeta in enumerate(zetas):
        text = ''
        for eta in etas:
            df_ = df[df['zeta'] == zeta]
            df_ = df_[df_['eta'] == eta]
            sat = df_['unsat'].sum()
            unsat = df_['sat'].sum()
            timeout = df_['timeout'].sum()
            text += f' & {sat/10}/{unsat/10}/{timeout/10}'
        line = pres[i] + text + posts[i]
        print(line)
