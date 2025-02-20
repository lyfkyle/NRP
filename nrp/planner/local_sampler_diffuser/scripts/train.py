import diffuser.utils as utils
import torch
from path_dataset import MyDataset

def main(**deps):
    from ml_logger import logger, RUN
    # from config.locomotion_config import Config
    from config.wbmp8dof_config import Config

    RUN._update(deps)
    Config._update(deps)

    # logger.remove('*.pkl')
    # logger.remove("traceback.err")
    logger.log_params(Config=vars(Config), RUN=vars(RUN))
    logger.log_text("""
                    charts:
                    - yKey: loss
                      xKey: steps
                    - yKey: a0_loss
                      xKey: steps
                    """, filename=".charts.yml", dedent=True, overwrite=True)

    torch.backends.cudnn.benchmark = True
    utils.set_seed(Config.seed)
    # -----------------------------------------------------------------------------#
    # ---------------------------------- dataset ----------------------------------#
    # -----------------------------------------------------------------------------#

    # dataset_config = utils.Config(
    #     Config.loader,
    #     savepath='dataset_config.pkl',
    #     env=Config.dataset,
    #     horizon=Config.horizon,
    #     normalizer=Config.normalizer,
    #     preprocess_fns=Config.preprocess_fns,
    #     use_padding=Config.use_padding,
    #     max_path_length=Config.max_path_length,
    #     include_returns=Config.include_returns,
    #     returns_scale=Config.returns_scale,
    #     discount=Config.discount,
    #     termination_penalty=Config.termination_penalty,
    # )

    # dataset_config = utils.Config(
    #     Config.loader,
    #     savepath='dataset_config.pkl',
    #     data_dir=Config.data_dir,
    #     dataset_size=Config.dataset_size,
    #     device=Config.device,
    #     occ_grid_dim=40
    # )

    # render_config = utils.Config(
    #     Config.renderer,
    #     savepath='render_config.pkl',
    #     env=Config.dataset,
    # )

    # dataset = dataset_config()
    dataset = MyDataset(
        data_dir=Config.data_dir,
        dataset_size=Config.dataset_size,
        device=Config.device,
        occ_grid_dim=40
    )
    # renderer = render_config()
    observation_dim = 8
    action_dim = 0

    # -----------------------------------------------------------------------------#
    # ------------------------------ model & trainer ------------------------------#
    # -----------------------------------------------------------------------------#
    model_config = utils.Config(
        Config.model,
        savepath='model_config.pkl',
        horizon=Config.horizon,
        transition_dim=observation_dim + action_dim,
        cond_dim=1608,
        collision_dim=Config.collision_dim,
        dim_mults=Config.dim_mults,
        returns_condition=Config.returns_condition,
        dim=Config.dim,
        condition_dropout=Config.condition_dropout,
        calc_energy=Config.calc_energy,
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
        ## loss weighting
        action_weight=Config.action_weight,
        loss_weights=Config.loss_weights,
        loss_discount=Config.loss_discount,
        returns_condition=Config.returns_condition,
        condition_guidance_w=Config.condition_guidance_w,
        device=Config.device,
        inference=False,
    )

    trainer_config = utils.Config(
        utils.Trainer,
        savepath='trainer_config.pkl',
        train_batch_size=Config.batch_size,
        train_lr=Config.learning_rate,
        gradient_accumulate_every=Config.gradient_accumulate_every,
        ema_decay=Config.ema_decay,
        sample_freq=Config.sample_freq,
        save_freq=Config.save_freq,
        log_freq=Config.log_freq,
        label_freq=int(Config.n_train_steps // Config.n_saves),
        save_parallel=Config.save_parallel,
        bucket=Config.bucket,
        n_reference=Config.n_reference,
        train_device=Config.device,
        save_checkpoints=Config.save_checkpoints,
    )

    # -----------------------------------------------------------------------------#
    # -------------------------------- instantiate --------------------------------#
    # -----------------------------------------------------------------------------#

    model = model_config()

    diffusion = diffusion_config(model)

    trainer = trainer_config(diffusion, dataset, None)

    # -----------------------------------------------------------------------------#
    # ------------------------ test forward & backward pass -----------------------#
    # -----------------------------------------------------------------------------#

    utils.report_parameters(model)

    logger.print('Testing forward...', end=' ', flush=True)
    batch = utils.batchify(dataset[0], Config.device)
    print(*batch[0].shape)
    loss, _ = diffusion.loss(*batch)
    loss.backward()
    logger.print('✓')

    # -----------------------------------------------------------------------------#
    # --------------------------------- main loop ---------------------------------#
    # -----------------------------------------------------------------------------#

    n_epochs = int(Config.n_train_steps // Config.n_steps_per_epoch)

    # checkpoint = 250000
    # trainer.load(checkpoint)
    # for i in range(checkpoint // Config.n_steps_per_epoch, n_epochs):
    for i in range(n_epochs):
        logger.print(f'Epoch {i} / {n_epochs} | {logger.prefix}')
        trainer.train(n_train_steps=Config.n_steps_per_epoch)


if __name__ == '__main__':
    kwargs = {
        'RUN.prefix': 'diffuser/test',
        'seed': 100,
        # 'predict_epsilon': True,
        # 'condition_dropout': 0.25,
        # 'diffusion': 'models.GaussianDiffusion'
    }

    main(**kwargs)