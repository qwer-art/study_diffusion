import os.path as osp
import sys

import numpy as np
import torch

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from ddpm.network import *
from ddpm.dataset import get_definite_dataloader
from ddpm.ddpm_model import DDPM
import matplotlib.pyplot as plt

configs = [
    convnet_small_cfg, convnet_medium_cfg, convnet_big_cfg, unet_1_cfg,
    unet_res_cfg
]

def show_batch_image(batch_image):
    # n,c,h,w->c,h,n,w
    n, c, h, w = batch_image.shape
    batch_image = batch_image.permute(1, 2, 0, 3)
    batch_image = batch_image.contiguous().view(c, h, n * w)
    print(f"batch_image: {batch_image.shape}")
    # c,h,nw
    batch_image = batch_image.to('cpu').numpy()
    batch_image = np.transpose(batch_image, (1, 2, 0))

    plt.imshow(batch_image)
    plt.axis('off')
    plt.show()

def main():
    device = "cuda"
    ### dataset
    batch_size = 1
    dataloader = get_definite_dataloader(batch_size)
    ### model
    n_steps = 1000
    config_id = 4
    model_path = 'model_param/model_unet_res.pth'
    config = configs[config_id]
    net = build_network(config, n_steps)
    net.load_state_dict(torch.load(model_path))
    net = net.to(device)
    ### ddpm
    ddpm = DDPM(device, n_steps)

    ## 可视化网络结构
    data_iter = iter(dataloader)
    # x
    x, _ = next(data_iter)
    x = x.to(device)
    # t
    t = torch.tensor([513]).to(device)
    xt = ddpm.sample_forward(x, t)
    if True:
        batch_image = torch.cat([x, xt], dim=0)
        show_batch_image(batch_image)
    ### xo->xt


    ### xt->xo

def test():
    n_steps = 1000
    d_model = 128

    pe = PositionalEncoding(1000,128)
    t = pe(torch.tensor([512]))
    print(f"t: {t.shape}")


if __name__ == '__main__':
    # main()
    test()