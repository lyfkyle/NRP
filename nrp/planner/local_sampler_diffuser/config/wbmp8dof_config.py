import torch
import os.path as osp
import math

from params_proto.neo_proto import ParamsProto, PrefixProto, Proto

CUR_DIR = osp.dirname(osp.abspath(__file__))

class Config(ParamsProto):
    # misc
    seed = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bucket = '/home/yunfan/whole-body-motion-planning/src/wbmp8dof/local_sampler_diffuser/weights'
    dataset = 'hopper-medium-expert-v2'

    ## model
    model = 'models.TemporalUnet'
    # model = 'models.MLPnet'
    # diffusion = 'models.GaussianInvDynDiffusion'  # We don't need inverse dynamcis as we do not have actions
    diffusion = 'models.GaussianDiffusion'
    horizon = 20
    n_diffusion_steps = 50  # TODO: try smaller
    action_weight = 0  # no action
    loss_weights = None
    loss_discount = 1
    predict_epsilon = True  # directly predict x_0
    dim_mults = (1, 1, 1)  # TODO: try (1, 2, 4)
    returns_condition = True
    calc_energy=False
    dim=64
    collision_dim=1
    condition_dropout=0.25
    condition_guidance_w = 1.2
    test_ret=0.9
    renderer = 'utils.MuJoCoRenderer'  # Might not need
    hidden_dim = 256
    ar_inv = False
    train_only_inv = False

    ## dataset
    loader = 'path_dataset.MyDataset'
    # data_dir = osp.join(CUR_DIR, "../dataset/k_shortest_train")
    data_dir = osp.join(CUR_DIR, "../dataset/mixed_train")
    dataset_size = 403152 # 296388 # 186388
    # test_data_dir = osp.join(CUR_DIR, "../dataset/k_shortest_test")
    test_data_dir = osp.join(CUR_DIR, "../dataset/mixed_test")
    test_dataset_size = 1969 # 1490 # 940
    # joint_bounds = torch.asarray([2.0, 2.0, math.radians(180), math.radians(180), math.radians(180), math.radians(180), math.radians(180), math.radians(180)])

    normalizer = 'CDFNormalizer'
    preprocess_fns = []
    clip_denoised = True
    use_padding = False  # trajectoy cannot be padded
    include_returns = True
    discount = 0.99
    max_path_length = 20
    termination_penalty = -100
    returns_scale = 400.0 # Determined using rewards from the dataset

    ## training
    n_steps_per_epoch = 10000
    loss_type = 'l2'
    n_train_steps = 5e5
    batch_size = 32
    learning_rate = 2e-4
    gradient_accumulate_every = 2
    ema_decay = 0.995
    log_freq = 1000
    save_freq = 50000
    sample_freq = 25000
    n_saves = 5
    save_parallel = False
    n_reference = 8
    save_checkpoints = True
