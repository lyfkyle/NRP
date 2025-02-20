import diffuser.utils as utils
from ml_logger import logger
import torch
from copy import deepcopy
import numpy as np
import os
import time

# import gym
# from config.locomotion_config import Config
from diffuser.utils.arrays import to_torch, to_np, to_device
# from diffuser.datasets.d4rl import suppress_output

from path_dataset import MyDataset
from torch.utils.data import DataLoader
from utils import visualize_nodes_local, visualize_diffusion, visualize_diffusion_with_guidance

BATCHSIZE = 10

def evaluate(**deps):
    from ml_logger import logger, RUN
    # from config.locomotion_config import Config
    from config.wbmp8dof_config import Config

    RUN._update(deps)
    Config._update(deps)

    logger.remove('*.pkl')
    logger.remove("traceback.err")
    logger.log_params(Config=vars(Config), RUN=vars(RUN))

    # Config.device = 'cuda'

    if Config.predict_epsilon:
        prefix = f'predict_epsilon_{Config.n_diffusion_steps}_1000000.0'
    else:
        prefix = f'predict_x0_{Config.n_diffusion_steps}_1000000.0'

    # Load configs
    torch.backends.cudnn.benchmark = True
    utils.set_seed(Config.seed)

    # observation_dim = dataset.observation_dim
    # action_dim = dataset.action_dim

    observation_dim = 8
    action_dim = 0

    if Config.diffusion == 'models.GaussianInvDynDiffusion':
        transition_dim = observation_dim
    else:
        transition_dim = observation_dim + action_dim

    model_config = utils.Config(
        Config.model,
        savepath='model_config.pkl',
        horizon=Config.horizon,
        transition_dim=transition_dim,
        cond_dim=observation_dim,
        collision_dim=Config.collision_dim,
        dim_mults=Config.dim_mults,
        dim=Config.dim,
        returns_condition=Config.returns_condition,
        device=Config.device,
    )

    diffusion_config = utils.Config(
        Config.diffusion,
        savepath='diffusion_config.pkl',
        horizon=Config.horizon,
        observation_dim=observation_dim,
        action_dim=action_dim,
        n_timesteps=Config.n_diffusion_steps,
        loss_type=Config.loss_type,
        clip_denoised=Config.clip_denoised,
        predict_epsilon=Config.predict_epsilon,
        # hidden_dim=Config.hidden_dim,
        ## loss weighting
        action_weight=Config.action_weight,
        loss_weights=Config.loss_weights,
        loss_discount=Config.loss_discount,
        returns_condition=Config.returns_condition,
        device=Config.device,
        condition_guidance_w=Config.condition_guidance_w,
        inference=True,
    )

    # -----------------------------------------------------------------------------#
    # -------------------------------- instantiate --------------------------------#
    # -----------------------------------------------------------------------------#

    model = model_config()
    diffusion = diffusion_config(model)

    logger.print(utils.report_parameters(model), color='green')

    loadpath = os.path.join(Config.bucket, logger.prefix, 'checkpoint')
    # if Config.save_checkpoints:
    #     loadpath = os.path.join(loadpath, f'state_{step}.pt')
    # else:
    filepath = os.path.join(loadpath, 'state_500000.pt')
    state_dict = torch.load(filepath, map_location=Config.device)


    diffusion.load_state_dict(state_dict['ema'])

    pytorch_total_params = sum(p.numel() for p in diffusion.parameters())
    print("PARAM LEN:", pytorch_total_params)

    device = Config.device

    # Warm up
    start_time = time.time()
    for _ in range(2):
        conditions = {0: torch.zeros((10, 8), device=device)}
        occ_grid = torch.zeros((10, 1, 40, 40), device=device)
        goal_pos = torch.zeros((10, 9), device=device)
        collision = torch.zeros((BATCHSIZE, Config.collision_dim), device=device)
        samples = diffusion.conditional_sample(conditions, occ_grid, goal_pos, collision)
    end_time = time.time()
    print("warmup takes {}".format(end_time - start_time))

    dataset = MyDataset(
        data_dir=Config.test_data_dir,
        dataset_size=Config.test_dataset_size,
        device=Config.device,
        occ_grid_dim=40
    )

    # dataset = MyDataset(
    #     data_dir=Config.data_dir,
    #     dataset_size=Config.dataset_size,
    #     device=Config.device,
    #     occ_grid_dim=40
    # )

    dataloader = DataLoader(dataset, batch_size=BATCHSIZE, shuffle=True, drop_last=True, num_workers=10, pin_memory=True)

    joint_bounds = dataset.joint_bounds.to(device)

    for batch in dataloader:
        # print(batch)
        ## repeat each item in conditions `n_samples` times
        cond = batch[1]
        # cond[0] = batch[1][0].repeat(BATCHSIZE, 1)
        # occ_grid = to_device(batch[2].repeat(BATCHSIZE, 1, 1, 1), device)
        # goal_pos_normed = to_device(batch[3].repeat(BATCHSIZE, 1), device)
        conditions = to_device(cond, device)
        occ_grid = to_device(batch[2], device)
        goal_pos_normed = to_device(batch[3], device)
        collision = torch.ones((BATCHSIZE, Config.collision_dim), device=device)
        start_time = time.time()
        diffusion.return_condition = True
        diffusion.condition_guidance_w = Config.condition_guidance_w
        samples, diffusion_process = diffusion.conditional_sample(conditions, occ_grid, goal_pos_normed, collision, return_diffusion=True)
        end_time = time.time()
        print("diffusion process takes {}".format(end_time - start_time))
        # samples = to_np(samples)
        diffusion_process = to_np(diffusion_process * joint_bounds)
        print(samples.shape)
        print(diffusion_process.shape)

        # Get unconditioned process
        diffusion.condition_guidance_w = 0
        samples, uncond_process = diffusion.conditional_sample(conditions, occ_grid, goal_pos_normed, collision, return_diffusion=True)
        uncond_process = to_np(uncond_process * joint_bounds)
        diffusion.return_condition = False
        samples, cond_process = diffusion.conditional_sample(conditions, occ_grid, goal_pos_normed, collision, return_diffusion=True)
        cond_process = to_np(cond_process * joint_bounds)

        ## [ n_samples x horizon x (action_dim + observation_dim) ]
        for i in range(BATCHSIZE):  # BATCHSIZE

            local_occ_grid = to_np(occ_grid[i, :]).reshape([40, 40])
            path = to_np(samples[i, :, :] * joint_bounds)
            start_pos = to_np(conditions[0][i, :].flatten() * joint_bounds)
            goal_pos = to_np(goal_pos_normed[i, :8].flatten() * joint_bounds)
            file_name = os.path.join(loadpath, "test_output_{}_free1.png".format(i))
            # visualize_nodes_local(local_occ_grid, path, start_pos, goal_pos, show=False, save=True, file_name=file_name)
            # visualize_diffusion(local_occ_grid, diffusion_process[i], start_pos, goal_pos, show=False, save=True, file_name=os.path.join(loadpath, "test_output_{}.gif".format(i)))
            visualize_diffusion_with_guidance(local_occ_grid, diffusion_process[i], uncond_process[i], cond_process[i], start_pos, goal_pos, show=False, save=True, file_name=os.path.join(loadpath, "test_output_{}_guidance.gif".format(i)))            
            # break
        break

if __name__ == '__main__':
    kwargs = {
        'RUN.prefix': 'diffuser/test',
        'seed': 100,
        # 'predict_epsilon': True,
        # 'condition_dropout': 0.25,
        # 'diffusion': 'models.GaussianDiffusion'
    }

    evaluate(**kwargs)