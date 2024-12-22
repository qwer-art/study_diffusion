import os.path as osp
import sys
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from ddpm.network import *
from ddpm.dataset import get_definite_dataloader
from ddpm.ddpm import DDPM
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

configs = [
    convnet_small_cfg, convnet_medium_cfg, convnet_big_cfg, unet_1_cfg,
    unet_res_cfg
]

if __name__ == '__main__':
    ### dataset
    batch_size = 4
    dataloader = get_definite_dataloader(batch_size)
    ### model
    n_steps = 1000
    config_id = 4
    model_path = 'model_unet_res.pth'
    config = configs[config_id]
    net = build_network(config, n_steps)
    ### ddpm
    device = "cuda"
    ddpm = DDPM(device, n_steps)
    ### writer
    ## 可视化网络结构
    writer = SummaryWriter(log_dir="../run/version1")
    data_iter = iter(dataloader)
    images,labels = next(data_iter)
    print(f"images: {images.shape},labels: {labels.shape}")


