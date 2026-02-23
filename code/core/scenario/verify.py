import pandas as pd
import os

def verify_one(onnx, vnnlib, timeout, args):
    os.chdir('neuralsat/src')
    onnx1 = os.path.join('..', '..', onnx)
    vnnlib1 = os.path.join('..', '..', vnnlib)
    path = '../../../data/csv/result.txt'
    os.system(f'rm -rf "{path}"')
    cmd  = f'python3 -W ignore main.py --disable_restart --verbosity=2'
    cmd += f' --net={onnx1} --spec={vnnlib1}'
    cmd += f' --result_file={path} --timeout={timeout}'
    if not args.verbose:
        cmd += f' > /dev/null 2>&1'
    os.system(cmd)
    sat = unsat = timeout = 0
    if os.path.exists(path):
        status = open(path).read().strip().split(',')[0]
        # print(f'[neuralsat] {status=}')
        if status == 'sat':
            sat = 1
        elif status == 'unsat':
            unsat = 1
        elif status == 'timeout':
            timeout = 1
        else:
            NotImplementedError
    else:
        timeout = 1
    os.chdir(args.core_dir)
    return sat, unsat, timeout

def verify(args):
    # onnx
    onnx = f'../data/onnx/{args.dataset}.onnx'
    # timeout
    timeout = 60
    # list of properties
    v_output_types = ['over', 'under']
    etas = [0.01, 0.05, 0.1, 0.15, 0.2]
    zetas = {
        'over': [0.01, 0.05, 0.1],
        'under': [0.01, 0.05, 0.1],
    }
    info = {
        'v_output_type': [],
        'eta': [],
        'zeta': [],
        'i': [],
        'unsat': [],
        'sat': [],
        'timeout': [],
    }
    for i in range(args.n_v_setup):
        for v_output_type in v_output_types:
            for eta in etas:
                for zeta in zetas[v_output_type]:
                    # path
                    vnnlib = os.path.join(args.vnnlib_dir, f'{args.dataset}_{eta}_{v_output_type}_{zeta}_0.vnnlib')
                    sat, unsat, timeout = verify_one(onnx, vnnlib, timeout, args)
                    # report
                    info['v_output_type'].append(v_output_type)
                    info['eta'].append(eta)
                    info['zeta'].append(zeta)
                    info['i'].append(i)
                    info['unsat'].append(unsat)
                    info['sat'].append(sat)
                    info['timeout'].append(timeout)
                    print(f'{i=} {eta=} {v_output_type=} {zeta=} {unsat=} {sat=} {timeout=}')
                    # export csv
                    df = pd.DataFrame(info)
                    path = os.path.join(args.csv_dir, 'verify.csv')
                    df.to_csv(path, index=None)
