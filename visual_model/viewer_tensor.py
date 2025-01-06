import cv2
import einops
import numpy as np
import torch
import torch.nn as nn

def tensor_to_images(imgs):
    if (3 == len(imgs.shape) and 3 == imgs.shape[0]):
        imgs = einops.rearrange(imgs, 'c h w -> h w c')
    elif (3 == len(imgs.shape) and 3 != imgs.shape[0]):
        imgs = imgs.unsqueeze(1)  # 通过 unsqueeze 扩展成 (12, 1, 28, 28)
        imgs = imgs.expand(-1, 3, -1, -1)
        n = imgs.shape[0]
        b1 = int(n ** 0.5)
        b2 = n // b1
        imgs = einops.rearrange(imgs, '(b1 b2) c h w -> (b1 h) (b2 w) c', b1=b1, b2=b2)
    elif (4 == len(imgs.shape)):
        n = imgs.shape[0]
        b1 = int(n ** 0.5)
        b2 = n // b1
        imgs = einops.rearrange(imgs, '(b1 b2) c h w -> (b1 h) (b2 w) c', b1=b1, b2=b2)
    else:
        print(f"[ Fail ],shape_size: {imgs.shape}")

    imgs = (imgs + 1) / 2 * 255
    imgs = imgs.clamp(0, 255)
    imgs = imgs.numpy().astype(np.uint8)
    return imgs

if __name__ == '__main__':
    img_shape = (81, 3, 28, 28)
    imgs = torch.randn(img_shape)
    imgs = tensor_to_images(imgs)

    cv2.imshow("imgs", imgs)
    cv2.waitKey(-1)
