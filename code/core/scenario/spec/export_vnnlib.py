import numpy as np
import simulator
import torch
import os

from beartype import beartype
import torch

def write_vnnlib_over(spec_path, x_lb, x_ub, y_thres, args):
    # input bounds
    x_lb = x_lb.flatten()
    x_ub = x_ub.flatten()
    n_class = 1
    with open(spec_path, "w") as f:
        f.write(f"; Specification for {spec_path}\n")

        f.write(f"\n; Definition of input variables\n")
        for i in range(len(x_ub)):
            f.write(f"(declare-const X_{i} Real)\n")

        f.write(f"\n; Definition of output variables\n")
        for i in range(n_class):
            f.write(f"(declare-const Y_{i} Real)\n")

        f.write(f"\n; Definition of input constraints\n")
        for i in range(len(x_ub)):
            f.write(f"(assert (<= X_{i} {x_ub[i]:.8f}))\n")
            f.write(f"(assert (>= X_{i} {x_lb[i]:.8f}))\n\n")

        f.write(f"\n; Definition of output constraints\n")
        f.write(f"(assert (>= Y_0 {y_thres[0].item()}))\n")
    return spec_path

def write_vnnlib_under(spec_path, x_lb, x_ub, y_thres, args):
    # input bounds
    x_lb = x_lb.flatten()
    x_ub = x_ub.flatten()
    n_class = 1
    with open(spec_path, "w") as f:
        f.write(f"; Specification for threshold {y_thres}\n")

        f.write(f"\n; Definition of input variables\n")
        for i in range(len(x_ub)):
            f.write(f"(declare-const X_{i} Real)\n")

        f.write(f"\n; Definition of output variables\n")
        for i in range(n_class):
            f.write(f"(declare-const Y_{i} Real)\n")

        f.write(f"\n; Definition of input constraints\n")
        for i in range(len(x_ub)):
            f.write(f"(assert (<= X_{i} {x_ub[i]:.8f}))\n")
            f.write(f"(assert (>= X_{i} {x_lb[i]:.8f}))\n\n")

        f.write(f"\n; Definition of output constraints\n")
        f.write(f"(assert (<= Y_0 {y_thres[0].item()}))\n")
    return spec_path

def export_vnnlib(args):
    # create test loader
    args.batch_size = 1
    _, loader = simulator.dataloader.create(args)
    #
    v_output_types = ['over', 'under']
    etas = [0.01, 0.05, 0.1, 0.15, 0.2]
    zetas = {
        'over': [0.01, 0.05, 0.1],
        'under': [0.01, 0.05, 0.1],
    }
    #
    for i, (x, y) in enumerate(loader):
        print(i)
        x = x.cpu()
        assert x.shape[0] == 1
        y = y.cpu()
        assert y.shape[0] == 1
        for v_output_type in v_output_types:
            for eta in etas:
                for zeta in zetas[v_output_type]:
                    # path
                    spec_path = os.path.join(args.vnnlib_dir, f'{args.dataset}_{eta}_{v_output_type}_{zeta}_{i}.vnnlib')
                    print(spec_path)
                    # input
                    x_lb = x
                    x_ub = x * (1 + eta)
                    if v_output_type == 'over':
                        # ouput
                        y_thres = y * (1 + zeta)
                        # write
                        write_vnnlib_over(spec_path, x_lb, x_ub, y_thres, args)
                    else:
                        # ouput
                        y_thres = y * (1 - zeta)
                        # write
                        write_vnnlib_under(spec_path, x_lb, x_ub, y_thres, args)
                    # exit()
        # stoping condition
        if i + 1 >= args.n_v_setup:
            break
