import torchvision
import torchvision.transforms as transforms

# 定义数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 指定下载源
torchvision.datasets.FashionMNIST.resources = [
    ('https://mirrors.aliyun.com/pytorch-datasets/fashion-mnist/train-images-idx3-ubyte.gz', '8d4fb7e6c68d591d4c3dfef9ec88bf0d'),
    ('https://mirrors.aliyun.com/pytorch-datasets/fashion-mnist/train-labels-idx1-ubyte.gz', '25c81989df183df01b3e8a0aad5dffbe'),
    ('https://mirrors.aliyun.com/pytorch-datasets/fashion-mnist/t10k-images-idx3-ubyte.gz', 'bef4ecab320f06d8554ea6380940ec79'),
    ('https://mirrors.aliyun.com/pytorch-datasets/fashion-mnist/t10k-labels-idx1-ubyte.gz', 'bb300cfdad3c16e7a12a480ee83cd310')
]

# 下载并加载训练集
trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                             download=True, transform=transform)