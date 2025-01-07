import os.path as osp
import sys
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

from ddpm.ddpm_model import *
import torch
from ddpm.network import (build_network, convnet_big_cfg,
                                  convnet_medium_cfg, convnet_small_cfg,
                                  unet_1_cfg, unet_res_cfg)

def forward_image():
    device = 'cuda'
    batch_size = 81
    n_steps = 1000

    dataloader = get_definite_dataloader(batch_size)
    x, _ = next(iter(dataloader))
    x = x.to(device)
    ddpm = DDPM(device, n_steps)
    for t in range(n_steps):
        if (0 == (t % 100)):
            image_path = f"image/forward/{t}.jpg"
            t = torch.tensor([t])
            t = t.unsqueeze(1)
            x_t = ddpm.sample_forward(x, t)
            image = tensor_to_image(x_t.cpu())
            cv2.imwrite(image_path, image)


def forward_image_300():
    device = 'cuda'
    batch_size = 81
    n_steps = 1000
    roi_steps = 300

    dataloader = get_definite_dataloader(batch_size)
    x, _ = next(iter(dataloader))
    x = x.to(device)
    ddpm = DDPM(device, n_steps)
    for t in range(roi_steps):
        if (0 == (t % 30)):
            image_path = f"image/forward_300/{t}.jpg"
            t = torch.tensor([t])
            t = t.unsqueeze(1)
            x_t = ddpm.sample_forward(x, t)
            image = tensor_to_image(x_t.cpu())
            cv2.imwrite(image_path, image)


configs = [
    convnet_small_cfg, convnet_medium_cfg, convnet_big_cfg, unet_1_cfg,
    unet_res_cfg
]
def backward_image():
    ### param
    device = 'cuda'
    batch_size = 81
    n_steps = 1000

    ### net
    model_path = 'model_param/model_unet_res.pth'
    config_id = 4
    config = configs[config_id]
    net = build_network(config, n_steps)
    net.load_state_dict(torch.load(model_path))


    ### x_t
    dataloader = get_definite_dataloader(batch_size)
    x, _ = next(iter(dataloader))
    x = x.to(device)
    t = torch.tensor([n_steps - 1])
    t = t.unsqueeze(1)
    ddpm = DDPM(device, n_steps)
    x_t = ddpm.sample_forward(x, t)

    ### backward
    ddpm.debug_backward(x_t,net,device)

def backward_image_300():
    ### param
    device = 'cuda'
    batch_size = 81
    n_steps = 1000
    roi_steps = 300

    ### net
    model_path = 'model_param/model_unet_res.pth'
    config_id = 4
    config = configs[config_id]
    net = build_network(config, n_steps)
    net.load_state_dict(torch.load(model_path))

    ### x_t
    dataloader = get_definite_dataloader(batch_size)
    x, _ = next(iter(dataloader))
    x = x.to(device)
    t = torch.tensor([roi_steps - 1])
    t = t.unsqueeze(1)
    ddpm = DDPM(device, n_steps)
    x_t = ddpm.sample_forward(x, t)

    ### backward
    ddpm.debug_backward_300(roi_steps, x_t, net, device)

if __name__ == '__main__':
    # backward_image()
    # forward_image_300()
    backward_image_300()