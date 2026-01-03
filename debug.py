import os
import os.path as osp
import sys
sys.path.append(osp.dirname(osp.abspath(__file__)))

import torch
from ddpm.ddpm_model import DDPM
from ddpm.network import build_network, unet_res_cfg
import cv2
import einops
import numpy as np


def tensor_to_image(imgs_tensor):
    """将tensor转换为图像用于保存"""
    if len(imgs_tensor.shape) == 4:
        n = imgs_tensor.shape[0]
        b1 = int(n ** 0.5)
        b2 = n // b1
        imgs_tensor = einops.rearrange(imgs_tensor, '(b1 b2) c h w -> (b1 h) (b2 w) c', b1=b1, b2=b2)
    else:
        print(f"[Warning] Unexpected shape: {imgs_tensor.shape}")

    imgs_tensor = (imgs_tensor + 1) / 2 * 255
    imgs_tensor = imgs_tensor.clamp(0, 255)
    imgs_tensor = imgs_tensor.numpy().astype(np.uint8)
    return imgs_tensor


def debug_diffusion_50_steps():
    """Debug: 输入纯高斯噪声，处理完整的1000步去噪过程，生成7x7网格展示"""
    # 配置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_steps = 1000  # 完整的1000步
    n_display = 49  # 显示49张图像 (7x7)
    n_samples = 1  # 只生成1张图像，追踪其去噪过程

    print(f"Using device: {device}")
    print(f"Running full {n_steps} steps reverse diffusion")
    print(f"Will save {n_display} intermediate results (7x7 grid)")

    # 创建DDPM模型
    ddpm = DDPM(device, n_steps)

    # 构建网络
    config = unet_res_cfg
    net = build_network(config, n_steps)

    # 加载预训练模型
    model_path = 'model_param/model_unet_res.pth'
    if osp.exists(model_path):
        print(f"Loading model from {model_path}")
        net.load_state_dict(torch.load(model_path, map_location=device))
        net = net.to(device)
        net.eval()
    else:
        print(f"Warning: Model file {model_path} not found!")
        print("Please train the model first or provide a valid model path.")
        return

    # 创建输出目录
    output_dir = 'debug_output'
    os.makedirs(output_dir, exist_ok=True)

    # 输入纯高斯噪声
    print("\n=== Step 1: Creating pure Gaussian noise ===")
    x = torch.randn(n_samples, 1, 28, 28).to(device)
    print(f"Initial noise shape: {x.shape}")
    print(f"Noise range: [{x.min():.3f}, {x.max():.3f}]")

    # 执行完整的1000步反向扩散，并保存中间结果
    print(f"\n=== Step 2: Running full {n_steps} steps reverse diffusion ===")

    # 计算保存间隔：从1000步中均匀选择49步
    save_steps = torch.linspace(n_steps - 1, 0, n_display).long().tolist()

    # 用于保存所有中间图像
    intermediate_images = []

    with torch.no_grad():
        for t in range(n_steps - 1, -1, -1):
            # 执行一步反向扩散
            x = ddpm.sample_backward_step(x, t, net, simple_var=True, clip_x0=True)

            # 如果当前步在需要保存的列表中
            if t in save_steps:
                intermediate_images.append(x.clone().cpu())
                position = save_steps.index(t) + 1
                print(f"Step {n_steps - t}/{n_steps} (t={t}): saved image {position}/{n_display}, range: [{x.min():.3f}, {x.max():.3f}]")

            torch.cuda.empty_cache()

    # 保存7x7网格的大图
    print(f"\n=== Step 3: Generating 7x7 grid ===")

    # 将所有中间图像拼接成一个7x7的网格
    # intermediate_images是49个(1, 1, 28, 28)的tensor
    grid = torch.cat(intermediate_images, dim=0)  # (49, 1, 28, 28)

    # 保存为7x7网格
    grid_img = tensor_to_image(grid)
    cv2.imwrite(f'{output_dir}/denoise_7x7_grid.png', grid_img)
    print(f"Saved 7x7 grid to {output_dir}/denoise_7x7_grid.png")

    # 同时保存初始噪声和最终结果
    initial_img = tensor_to_image(intermediate_images[0])
    cv2.imwrite(f'{output_dir}/step_0_noise.png', initial_img)
    print(f"Saved initial noise to {output_dir}/step_0_noise.png")

    final_img = tensor_to_image(intermediate_images[-1])
    cv2.imwrite(f'{output_dir}/step_1000_final.png', final_img)
    print(f"Saved final result to {output_dir}/step_1000_final.png")

    print(f"\nAll debug images saved to '{output_dir}/' directory")
    print(f"Total steps processed: {n_steps}")
    print(f"Displayed {n_display} intermediate steps in 7x7 grid")


if __name__ == '__main__':
    debug_diffusion_50_steps()