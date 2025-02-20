if __name__ == '__main__':
    from ml_logger import logger, instr, needs_relaunch
    from analysis import RUN
    import jaynes
    from scripts.train import main
    from config.locomotion_config import Config
    from params_proto.neo_hyper import Sweep

    sweep = Sweep(RUN, Config).load("default_inv.jsonl")

    '''
    kwargs = {
    'RUN.prefix':
    'diffuser/default_inv/predict_epsilon_200_1000000.0/dropout_0.25/hopper-medium-expert-v2/100'
    'seed':
    100
    'returns_condition':
    True
    'predict_epsilon':
    True
    'n_diffusion_steps':
    200
    'condition_dropout':
    0.25
    'diffusion':
    'models.GaussianInvDynDiffusion'
    }
    '''

    for kwargs in sweep:
        logger.print(RUN.prefix, color='green')
        jaynes.config("local")
        thunk = instr(main, **kwargs)
        jaynes.run(thunk)

    jaynes.listen()
