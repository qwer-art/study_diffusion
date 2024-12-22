import os.path as osp
import sys

from torch.cuda import device

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

def show_network(model,x,t):
    ## tensorboard --logdir="./run/version1"
    writer = SummaryWriter(log_dir="./run/version1")
    writer.add_graph(model, (x, t))
    writer.close()

def debug_model(model,x,t):
    eps = torch.randn_like(x).to(device)
    x_t = ddpm.sample_forward(x, t, eps)
    print(f"x: {x.shape},t: {t.shape},x_t: {x_t.shape}")
    eps_theta = model(x_t, t.reshape(current_batch_size, 1))
    print(f"eps_theta: {eps_theta.shape}")
    loss_fn = nn.MSELoss()
    loss = loss_fn(eps_theta, eps)
    print(f"loss: {loss}")

if __name__ == '__main__':
    device = "cuda"
    ### dataset
    batch_size = 4
    dataloader = get_definite_dataloader(batch_size)
    ### model
    n_steps = 1000
    config_id = 4
    model_path = 'model_unet_res.pth'
    config = configs[config_id]
    model = build_network(config, n_steps)
    model = model.to(device)
    ### ddpm
    ddpm = DDPM(device, n_steps)
    ### writer
    ## 可视化网络结构
    data_iter = iter(dataloader)
    # x
    x, _ = next(data_iter)
    x = x.to(device)
    # t
    current_batch_size = x.shape[0]
    t = torch.randint(0, n_steps, (current_batch_size,)).to(device)
    # visual
    # show_network(model, x, t)
    debug_model(model, x, t)
