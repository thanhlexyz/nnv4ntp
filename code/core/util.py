import argparse
import torch
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def create_folders(args):
    ls = [args.csv_dir, args.data_dir, args.figure_dir, args.model_dir, args.onnx_dir, args.vnnlib_dir]
    for folder in ls:
        os.makedirs(folder, exist_ok=True)

def set_default_device(args):
    torch.set_default_device(args.device)

base_folder = os.path.dirname(os.path.dirname(__file__))

def get_args():
    # create args parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, default='main')
    parser.add_argument('--mode', type=str, default='test')
    # dataset
    parser.add_argument('--dataset', type=str, default='telecomItalia') # opnet not supported yet
    # input
    parser.add_argument('--n_look_back', type=int, default=10) # calibration parameter for capacity forecasting
    # loss function
    parser.add_argument('--deepcog_alpha', type=float, default=1.0) # calibration parameter for capacity forecasting
    parser.add_argument('--deepcog_epsilon', type=float, default=0.1) # calibration parameter for capacity forecasting
    # training
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=int, default=3e-4)
    parser.add_argument('--n_train_epoch', type=int, default=50)
    parser.add_argument('--n_test_epoch', type=int, default=1)
    parser.add_argument('--n_save_epoch', type=int, default=1)
    # verification
    parser.add_argument('--n_v_setup', type=int, default=10)
    parser.add_argument('--v_output_type', type=str, default='over') # under (check for forecast over/under estimation)
    # data directory
    parser.add_argument('--vnnlib_dir', type=str, default='../data/vnnlib')
    parser.add_argument('--figure_dir', type=str, default='../data/figure')
    parser.add_argument('--model_dir', type=str, default='../data/model')
    parser.add_argument('--onnx_dir', type=str, default='../data/onnx')
    parser.add_argument('--data_dir', type=str, default='../data/data')
    parser.add_argument('--csv_dir', type=str, default='../data/csv')
    parser.add_argument('--core_dir', type=str, default=f'{ROOT_DIR}/core')
    # plot
    parser.add_argument('--metric', type=str, default='loss')
    parser.add_argument('--n_smooth', type=int, default=50)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--no_pbar', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    # other
    if torch.cuda.is_available():
        parser.add_argument('--device', type=str, default='cuda:0')
    else:
        parser.add_argument('--device', type=str, default='cpu')
    # parse args
    args = parser.parse_args()
    # create folders
    create_folders(args)
    # additional args
    return args
