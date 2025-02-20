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
from utils import visualize_nodes_local, visualize_diffusion, visualize_diffusion_with_guidance, FkTorch, calculate_stats

from diffuser.models.col_checker import ColCheckerSmall

BATCHSIZE = 50

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
    occ_grid_dim = 40

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
    # filepath = os.path.join(loadpath, 'state_500000.pt')
    # state_dict = torch.load(filepath, map_location=Config.device)

    # diffusion.load_state_dict(state_dict['ema'])

    pytorch_total_params = sum(p.numel() for p in diffusion.parameters())
    print("PARAM LEN:", pytorch_total_params)

    device = Config.device

    # Loading collision checker
    linkpos_dim = 12
    # col_checker_path = os.path.join(os.path.dirname(__file__), '../../local_sampler_d/models/model_col_final.pt')
    col_checker_path = os.path.join(os.path.dirname(__file__), '../weights/col_checker_best.pt')
    col_checker = ColCheckerSmall(observation_dim + linkpos_dim, occ_grid_size=occ_grid_dim)
    col_checker.load_state_dict(torch.load(col_checker_path, map_location=torch.device('cpu')))
    col_checker.to(device)
    col_checker.eval()

    fk = FkTorch(device)

    # Warm up
    start_time = time.time()
    for _ in range(0):
        conditions = {0: torch.zeros((10, observation_dim), device=device)}
        occ_grid = torch.zeros((10, 1, occ_grid_dim, occ_grid_dim), device=device)
        goal_pos = torch.zeros((10, 9), device=device)
        collision = torch.zeros((BATCHSIZE, Config.collision_dim), device=device)
        samples = diffusion.conditional_sample(conditions, occ_grid, goal_pos, collision)
    end_time = time.time()
    print("warmup takes {}".format(end_time - start_time))

    dataset = MyDataset(
        data_dir=Config.test_data_dir,
        dataset_size=Config.test_dataset_size,
        device=Config.device,
        occ_grid_dim=occ_grid_dim
    )

    # dataset = MyDataset(
    #     data_dir=Config.data_dir,
    #     dataset_size=Config.dataset_size,
    #     device=Config.device,
    #     occ_grid_dim=occ_grid_dim
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
        samples = to_device(batch[0], device) * joint_bounds
        occ_grid = to_device(batch[2], device)
        goal_pos_normed = to_device(batch[3], device)
        gt_collision = to_device(batch[4], device)
        
        print(samples.shape)
        sample_linkpos = fk.get_link_positions(samples.view(-1, observation_dim)).view(samples.shape[0], samples.shape[1], -1)
        samples_w_linkpos = torch.cat((samples, sample_linkpos), dim=2)
        checker_collision = col_checker.forward_trajectory(occ_grid, samples_w_linkpos)
        checker_collision = (checker_collision > 0.5).to(torch.float32)
        collision_diff = torch.sum((torch.square(checker_collision - gt_collision)), dim=1)
        
        checker_collision = to_np(checker_collision)
        gt_collision = to_np(gt_collision)
        # true_pos = np.sum(np.logical_and(checker_collision, gt_collision))
        # true_neg = np.sum(np.logical_and(1-checker_collision, 1-gt_collision))
        # false_pos = np.sum(np.logical_and(checker_collision, 1-gt_collision))
        # false_neg = np.sum(np.logical_and(1-checker_collision, gt_collision))
        checker_free = np.logical_not(checker_collision)
        gt_free = np.logical_not(gt_collision)
        true_pos = np.sum(np.logical_and(checker_free, gt_free))
        true_neg = np.sum(np.logical_and(1-checker_free, 1-gt_free))
        false_pos = np.sum(np.logical_and(checker_free, 1-gt_free))
        false_neg = np.sum(np.logical_and(1-checker_free, gt_free))
        print("true_pos:", true_pos)
        print("true_neg:", true_neg)
        print("false_pos:", false_pos)
        print("false_neg:", false_neg)

        accuracy, precision, recall = calculate_stats(true_pos, true_neg, false_pos, false_neg)
        print("accuracy:", accuracy)
        print("precision:", precision)
        print("recall:", recall)
        break
        # collision = torch.ones((BATCHSIZE, Config.collision_dim), device=device)
        # start_time = time.time()
        # diffusion.return_condition = True
        # diffusion.condition_guidance_w = Config.condition_guidance_w
        # samples, diffusion_process = diffusion.conditional_sample(conditions, occ_grid, goal_pos_normed, collision, return_diffusion=True)
        # end_time = time.time()
        # print("diffusion process takes {}".format(end_time - start_time))
        # # samples = to_np(samples)
        # diffusion_process = to_np(diffusion_process * joint_bounds)
        # print(samples.shape)
        # print(diffusion_process.shape)

        # # Get unconditioned process
        # diffusion.condition_guidance_w = 0
        # samples, uncond_process = diffusion.conditional_sample(conditions, occ_grid, goal_pos_normed, collision, return_diffusion=True)
        # uncond_process = to_np(uncond_process * joint_bounds)
        # diffusion.return_condition = False
        # samples, cond_process = diffusion.conditional_sample(conditions, occ_grid, goal_pos_normed, collision, return_diffusion=True)
        # cond_process = to_np(cond_process * joint_bounds)

        ## [ n_samples x horizon x (action_dim + observation_dim) ]
        for i in range(BATCHSIZE):  # BATCHSIZE
            print("ID:               ", i)
            print("gt_collision:     ", to_np(gt_collision[i]))
            print("checker_collision:", to_np(checker_collision[i]))
            print("diff:             ", to_np(collision_diff[i]) )
            local_occ_grid = to_np(occ_grid[i, :]).reshape([occ_grid_dim, occ_grid_dim])
            # path = to_np(samples[i, :, :] * joint_bounds)
            path = to_np(samples[i])
            start_pos = to_np(conditions[0][i, :].flatten() * joint_bounds)
            goal_pos = to_np(goal_pos_normed[i, :8].flatten() * joint_bounds)
            file_name = os.path.join(loadpath, "test_col_{}.png".format(i))
            # visualize_nodes_local(local_occ_grid, path, start_pos, goal_pos, show=False, save=True, file_name=file_name)
            # # visualize_diffusion(local_occ_grid, diffusion_process[i], start_pos, goal_pos, show=False, save=True, file_name=os.path.join(loadpath, "test_output_{}.gif".format(i)))
            # visualize_diffusion_with_guidance(local_occ_grid, diffusion_process[i], uncond_process[i], cond_process[i], start_pos, goal_pos, show=False, save=True, file_name=os.path.join(loadpath, "test_output_{}_guidance.gif".format(i)))            
            # # break        
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