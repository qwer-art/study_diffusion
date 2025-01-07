import os.path as osp
import sys
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

import matplotlib.pyplot as plt
import torch
import cv2
import einops
import numpy as np
from ddpm.dataset import get_definite_dataloader

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


class DDPM():

    def __init__(self,
                 device,
                 n_steps: int,
                 min_beta: float = 0.0001,
                 max_beta: float = 0.02):
        betas = torch.linspace(min_beta, max_beta, n_steps).to(device)
        alphas = 1 - betas
        alpha_bars = torch.empty_like(alphas)
        product = 1
        for i, alpha in enumerate(alphas):
            product *= alpha
            alpha_bars[i] = product
        self.betas = betas
        self.n_steps = n_steps
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        alpha_prev = torch.empty_like(alpha_bars)
        alpha_prev[1:] = alpha_bars[0:n_steps - 1]
        alpha_prev[0] = 1
        self.coef1 = torch.sqrt(alphas) * (1 - alpha_prev) / (1 - alpha_bars)
        self.coef2 = torch.sqrt(alpha_prev) * self.betas / (1 - alpha_bars)

    def sample_forward(self, x, t, eps=None):
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1)
        if eps is None:
            eps = torch.randn_like(x)
        res = eps * torch.sqrt(1 - alpha_bar) + torch.sqrt(alpha_bar) * x
        return res

    def debug_backward(self,x,net,device):
        simple_var = True,
        clip_x0 = True
        net = net.to(device)
        x = x.to(device)
        for t in range(self.n_steps - 1, -1, -1):
            with torch.no_grad():
                x = self.sample_backward_step(x, t, net, simple_var, clip_x0)
                if (0 == (t % 100)):
                    image_path = f"image/backward/{t}.jpg"
                    image = tensor_to_image(x.detach().cpu())
                    cv2.imwrite(image_path,image)
                torch.cuda.empty_cache()
        return x

    def sample_backward(self,
                        img_shape,
                        net,
                        device,
                        simple_var=True,
                        clip_x0=True):
        x = torch.randn(img_shape).to(device)
        net = net.to(device)
        for t in range(self.n_steps - 1, -1, -1):
            x = self.sample_backward_step(x, t, net, simple_var, clip_x0)
        return x

    def sample_backward_step(self, x_t, t, net, simple_var=True, clip_x0=True):

        n = x_t.shape[0]
        t_tensor = torch.tensor([t] * n,
                                dtype=torch.long).to(x_t.device).unsqueeze(1)
        eps = net(x_t, t_tensor)

        if t == 0:
            noise = 0
        else:
            if simple_var:
                var = self.betas[t]
            else:
                var = (1 - self.alpha_bars[t - 1]) / (
                    1 - self.alpha_bars[t]) * self.betas[t]
            noise = torch.randn_like(x_t)
            noise *= torch.sqrt(var)

        if clip_x0:
            x_0 = (x_t - torch.sqrt(1 - self.alpha_bars[t]) *
                   eps) / torch.sqrt(self.alpha_bars[t])
            x_0 = torch.clip(x_0, -1, 1)
            mean = self.coef1[t] * x_t + self.coef2[t] * x_0
        else:
            mean = (x_t -
                    (1 - self.alphas[t]) / torch.sqrt(1 - self.alpha_bars[t]) *
                    eps) / torch.sqrt(self.alphas[t])
        x_t = mean + noise

        return x_t


def visualize_forward():
    n_steps = 100
    device = 'cuda'
    dataloader = get_definite_dataloader(5)
    x, _ = next(iter(dataloader))
    x = x.to(device)
    ddpm = DDPM(device, n_steps)
    xts = []
    percents = torch.linspace(0, 0.99, 10)
    for percent in percents:
        t = torch.tensor([int(n_steps * percent)])
        t = t.unsqueeze(1)
        x_t = ddpm.sample_forward(x, t)
        xts.append(x_t)
    res = torch.stack(xts, 0)
    res = einops.rearrange(res, 'n1 n2 c h w -> (n2 h) (n1 w) c')
    res = (res.clip(-1, 1) + 1) / 2 * 255
    res = res.cpu().numpy().astype(np.uint8)

    # cv2.imwrite('workdirs/diffusion_forward_first_5.jpg', res)

def test_add_noise():
    n_steps = 100
    device = 'cuda'
    dataloader = get_definite_dataloader(5)
    x, _ = next(iter(dataloader))
    x = x.to(device)
    ddpm = DDPM(device, n_steps)

    percents = torch.linspace(0, 0.99, 10)
    for percent in percents:
        t = torch.tensor([int(n_steps * percent)])
        t = t.unsqueeze(1)

        x_t = ddpm.sample_forward(x, t)

        n, c, h, w = x.shape
        ## n,c,2h,w
        cx = torch.cat((x, x_t), dim=2)
        ## n,2h,w
        cx = cx.squeeze()
        ## n,w,2h
        cx = cx.permute(1, 0, 2)
        ## nw,2h
        cx = cx.flatten(1, 2)
        cx = cx.cpu().numpy()

        image_name = "diffusion_add_noise_" + str(int(n_steps * percent))
        image_path = f'workdirs/{image_name}.jpg'
        ### (2h,nw)
        plt.imshow(cx)
        plt.axis('off')
        plt.savefig(image_path)

def main():
    # visualize_forward()
    test_add_noise()

if __name__ == '__main__':
    main()