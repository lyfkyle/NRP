import utils
import torch
import os.path as osp

CUR_DIR = osp.dirname(osp.abspath(__file__))

decoder_model_folder = osp.join(CUR_DIR, "weights/general_mpt_panda_7d_trained/stage1/")
ar_model_folder = osp.join(CUR_DIR, "weights/general_mpt_panda_7d_trained/stage2/")

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
quantizer_model, decoder_model, context_env_encoder, ar_model = utils.get_inference_models(
    decoder_model_folder,
    ar_model_folder,
    device,
    n_e=2048,
    e_dim=8,
)
