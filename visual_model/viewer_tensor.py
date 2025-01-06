import os.path as osp
import sys
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

import cv2
import einops
import numpy as np
from ddpm.dataset import get_definite_dataloader
from ddpm.ddpm_model import DDPM
import torch

def tensor_to_image(imgs_tensor):
    if (3 == len(imgs_tensor.shape) and 3 == imgs_tensor.shape[0]):
        imgs_tensor = einops.rearrange(imgs_tensor, 'c h w -> h w c')
    elif (3 == len(imgs_tensor.shape) and 3 != imgs_tensor.shape[0]):
        imgs_tensor = imgs_tensor.unsqueeze(1)  # 通过 unsqueeze 扩展成 (12, 1, 28, 28)
        imgs_tensor = imgs_tensor.expand(-1, 3, -1, -1)
        n = imgs_tensor.shape[0]
        b1 = int(n ** 0.5)
        b2 = n // b1
        imgs_tensor = einops.rearrange(imgs_tensor, '(b1 b2) c h w -> (b1 h) (b2 w) c', b1=b1, b2=b2)
    elif (4 == len(imgs_tensor.shape)):
        n = imgs_tensor.shape[0]
        b1 = int(n ** 0.5)
        b2 = n // b1
        imgs_tensor = einops.rearrange(imgs_tensor, '(b1 b2) c h w -> (b1 h) (b2 w) c', b1=b1, b2=b2)
    else:
        print(f"[ Fail ],shape_size: {imgs_tensor.shape}")

    imgs_tensor = (imgs_tensor + 1) / 2 * 255
    imgs_tensor = imgs_tensor.clamp(0, 255)
    imgs_tensor = imgs_tensor.numpy().astype(np.uint8)
    return imgs_tensor


if __name__ == '__main__':
    device = 'cuda'
    batch_size = 81
    n_steps = 1000

    dataloader = get_definite_dataloader(batch_size)
    x, _ = next(iter(dataloader))
    x = x.to(device)
    ddpm = DDPM(device, n_steps)
    for t in range(n_steps):
        if (0 == (t % 100)):
            image_path = f"image_forward/{t}.jpg"
            t = torch.tensor([t])
            t = t.unsqueeze(1)
            print(f"t: {t},{t.shape}")
            x_t = ddpm.sample_forward(x, t)
            image = tensor_to_image(x_t.cpu())
            cv2.imwrite(image_path,image)