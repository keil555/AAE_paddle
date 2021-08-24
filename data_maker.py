from __future__ import print_function
import os 
import pickle
import numpy as np

import paddle
from paddle.io import random_split
from paddle.io import DataLoader
from paddle.vision import datasets
import paddle.vision.transforms as transforms
from utils.log import get_logger

# # Configure data loader
logger = get_logger('./logs/data_maker.log')
logger.info('start create datasets!')

os.makedirs("./model", exist_ok=True)
os.makedirs("./data", exist_ok=True)
os.makedirs("./images", exist_ok=True)
os.makedirs("./logs", exist_ok=True)

trainset = datasets.MNIST(
        mode="train", download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(), 
                # transforms.Normalize([0.5], [0.5])
            ]
        ),
    )
testset = datasets.MNIST(
        mode="test", download=True,
        transform=transforms.Compose(
            [transforms.ToTensor()]
        ),
    )
trainset, validset = random_split(trainset, [59000, 1000])

traindataloader = DataLoader(
    trainset, batch_size=100, shuffle=True,
)
validdataloader = DataLoader(
    validset, batch_size=1000, shuffle=True,
)
testdataloader = DataLoader(
    testset, batch_size=10000, shuffle=True,
)

valid_imgs, _ = next(iter(validdataloader))
test_imgs, _ = next(iter(testdataloader))
# valid_imgs = valid_imgs * 0.3081 + 0.1307

paddle.save(trainset, "./data/train")
paddle.save(valid_imgs, "./data/valid")
paddle.save(test_imgs, "./data/test")

logger.info('finish create datasets!')