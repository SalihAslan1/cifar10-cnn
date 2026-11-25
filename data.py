import numpy as np
import torch
from torchvision.datasets import CIFAR10

from model import CLASSES, ROOT

DATA_DIR = ROOT / "data"
DATASET_URL = "https://cave.cs.toronto.edu/kriz/cifar.html"
ARCHIVE_URL = "https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz"


class TorontoCIFAR10(CIFAR10):
    url = ARCHIVE_URL


def read_dataset(*, train, download=False):
    dataset = TorontoCIFAR10(str(DATA_DIR), train=train, download=download)
    return torch.from_numpy(dataset.data.copy()), torch.tensor(dataset.targets)


def stratified_split(labels, seed):
    rng = np.random.default_rng(seed)
    train, validation = [], []
    for label in range(len(CLASSES)):
        indices = rng.permutation(np.flatnonzero(labels.numpy() == label))
        validation.extend(indices[:500])
        train.extend(indices[500:])
    return torch.tensor(rng.permutation(train)), torch.tensor(rng.permutation(validation))


def channel_statistics(images):
    # Histogram counts give the exact moments without expanding the image tensor.
    means, stds = [], []
    values = np.arange(256) / 255
    for channel in range(3):
        counts = np.bincount(images[..., channel].numpy().ravel(), minlength=256)
        mean = (counts @ values) / counts.sum()
        variance = (counts @ (values - mean) ** 2) / counts.sum()
        means.append(mean)
        stds.append(np.sqrt(variance))
    return torch.tensor(means, dtype=torch.float32), torch.tensor(stds, dtype=torch.float32)
