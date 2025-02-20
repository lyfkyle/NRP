''' A script to save qunatized indexes.
'''

from modules.quantizer import VectorQuantizer
from modules.encoder import EncoderPreNorm

import numpy as np
import torch

import pickle
import os

from os import path as osp
from tqdm import tqdm

import argparse

from data_loader import q_max, q_min

CUR_DIR = osp.dirname(osp.abspath(__file__))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_dir', help="Directory where VQ-VAE model is kept", default='')
    parser.add_argument(
        '--data_dir', help="directory where data is stored", default='/root/data2d'
    )
    parser.add_argument(
        '--start_env', help="start environment of the model", type=int, default=0
    )
    parser.add_argument(
        '--samples', help="Number of envs to collect", type=int, default=25
    )
    parser.add_argument(
        '--save_dir', help="directory to save data"
    )
    parser.add_argument(
        '--env_type', help='train or val dataset', choices=['train', 'val']
    )
    parser.add_argument(
        '--robot', help='type of robot', choices=['2D', '6D', '7D', '8D', '11D', '14D']
    )
    parser.add_argument(
        '--num_keys', help="Number of dictionary elements", type=int, default=2048
    )
    
    args = parser.parse_args()

    if args.robot=='2D':
        c_space_dim=2
    if args.robot=='6D':
        c_space_dim=6
    if args.robot=='7D':
        c_space_dim=7
    if args.robot=='8D':
        c_space_dim=8
    if args.robot=='11D':
        c_space_dim=11
    if args.robot=='14D':
        c_space_dim=14

    model_args = dict(
        n_layers=3,
        n_heads=3,
        d_k=512,
        d_v=256,
        d_model=512,
        d_inner=1024,
        n_position=1000,
        dropout=0.1,
        c_space_dim=c_space_dim
    )

    device = 'cpu' if torch.cuda.is_available() else torch.device('cuda')

    encoder_model = EncoderPreNorm(**model_args)
    quantizer_model = VectorQuantizer(
        n_e=args.num_keys, e_dim=c_space_dim, latent_dim=model_args['d_model'])

    # model_dir = args.model_dir
    model_dir = osp.join(CUR_DIR, f"./models/{args.robot.lower()}/stage_1")
    checkpoint = torch.load(osp.join(model_dir, 'best_model.pkl'))
    
    # Load the state_dict
    encoder_model.load_state_dict(checkpoint['encoder_state'])
    quantizer_model.load_state_dict(checkpoint['quantizer_state'])

    for model in [encoder_model, quantizer_model]:
        model.eval()

    # data_dir = args.data_dir
    # save_dir = args.save_dir
    data_dir = osp.join(CUR_DIR, f"./dataset/{args.robot.lower()}")
    save_dir = osp.join(CUR_DIR, f"./dataset/{args.robot.lower()}_quant_key")

    for env_num in tqdm(range(args.start_env, args.start_env+args.samples)):
        env_dir = osp.join(data_dir, args.env_type, f'env_{env_num}')
        # Check if environment dir exists
        if not osp.isdir(env_dir):
            continue
        save_env_dir = osp.join(save_dir, args.env_type, f'env_{env_num}')
        # Check if folder exists, if not create one.
        if not osp.isdir(save_env_dir):
            os.mkdir(save_env_dir)

        path_list = [p for p in os.listdir(env_dir) if p[-2:] == '.p']
        for path_file in path_list:
            with open(osp.join(env_dir, path_file), 'rb') as f:
                data = pickle.load(f)
            
            if data['success']:
                if args.robot=='2D':
                    path_norm = data['path_interpolated']/24
                if args.robot=='6D':
                    path_norm = ((data['jointPath']-q_min)/(q_max-q_min))[:, :6]
                if args.robot=='8D':
                    path_norm = ((data['path_interpolated']-q_min)/(q_max-q_min))[:, :8]
                if args.robot=='11D':
                    path_norm = ((data['path_interpolated']-q_min)/(q_max-q_min))[:, :11]
                if args.robot=='14D':
                    q_bi_max = np.c_[q_max, q_max]
                    q_bi_min = np.c_[q_min, q_min]
                    path_norm = (data['path_interpolated']-q_bi_min)/(q_bi_max-q_bi_min)
                encoder_input = torch.as_tensor(path_norm, dtype=torch.float)[None, :].to(device)
                encoder_output, = encoder_model(encoder_input)
                _, (_, _, quant_keys) = quantizer_model(encoder_output, None)

                # NOTE: Gets unique keys but also sorts the result.
                # quant_keys = quant_keys.unique().numpy()
                # NOTE: Gets unique keys but doesn't sort them.
                # quant_keys = quant_keys.unique(sorted=False).flip(0).numpy()
                # NOTE: This removes redundant keys that r consecutive, and keeps the order
                # of the keys
                quant_keys = quant_keys.unique_consecutive().numpy()
                with open(osp.join(save_env_dir, path_file), 'wb') as f:
                    data = dict(keys=quant_keys)
                    pickle.dump(data, f)