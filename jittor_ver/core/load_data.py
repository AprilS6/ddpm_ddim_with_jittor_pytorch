import jittor as jt
from jittor.dataset.cifar import CIFAR10


def get_dataset(
        root: str="../../data",
        dataset_name: str = "cifar10",
        train: bool = True
    ):
    """
    下载CIFAR-10数据集，并返回标准化的Dataset
    Args:
        root: 数据集根目录
        dataset_name: 数据集名称
    Return:
        Dataset
    """
    if dataset_name == "cifar10":
        transform = jt.transform.Compose([
            jt.transform.RandomHorizontalFlip(), # 随机水平翻转，必须在ToTensor之前
            jt.transform.ToTensor(), # [0, 255] -> [0, 1]
            jt.transform.ImageNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [0, 1] -> [-1,1]
        ])
        dataset = CIFAR10(
            root=root,
            train=train,
            transform=transform
        )
    else:
        raise ValueError("Unsupported dataset name: {}".format(dataset_name))
    return dataset