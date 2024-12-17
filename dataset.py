import torchvision
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import cv2

def download_dataset():
    mnist = torchvision.datasets.MNIST(root='data/mnist', download=True)
    print('length of MNIST', len(mnist))
    id = 4
    img, label = mnist[id]
    print(f"label: {label}")

    image_name = f"image_id_{id}_label_{label}"
    image_path = f'workdirs/{image_name}.jpg'
    plt.imshow(img)  # squeeze() 用于移除单通道的维度
    plt.axis('off')  # 关闭坐标轴
    plt.savefig(image_path)  # 保存为 PNG 文件
    plt.close()
    # On computer with monitor
    # img.show()
    # img.save('work_dirs/tmp.jpg')
    # tensor = ToTensor()(img)
    # print(tensor.shape)
    # print(tensor.max())
    # print(tensor.min())

if __name__ == '__main__':
    download_dataset()