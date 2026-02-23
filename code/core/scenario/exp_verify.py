import os

from . import util

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

def exp_verify(args):
    for
