import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Lambda, ToTensor
import matplotlib.pyplot as plt

def download_dataset():
    mnist = torchvision.datasets.MNIST(root='data/mnist', download=True)
    print('length of MNIST', len(mnist))
    id = 4
    img, label = mnist[id]
    print(f"label: {label}")

    image_name = f"image_id_{id}_label_{label}"
    image_path = f'workdirs/{image_name}.jpg'
    # plt.imshow(img)  # squeeze() 用于移除单通道的维度
    # plt.axis('off')  # 关闭坐标轴
    # plt.savefig(image_path)  # 保存为 PNG 文件
    # plt.close()

def get_dataloader(batch_size: int):
    transform = Compose([ToTensor(), Lambda(lambda x: (x - 0.5) * 2)])
    dataset = torchvision.datasets.MNIST(root='data/mnist',
                                         transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def get_definite_dataloader(batch_size: int):
    transform = Compose([ToTensor(), Lambda(lambda x: (x - 0.5) * 2)])
    dataset = torchvision.datasets.MNIST(root='data/mnist',
                                         transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

def get_img_shape():
    return (1, 28, 28)

if __name__ == '__main__':
    download_dataset()