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
from config import args_parser

opt = args_parser()
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
            [transforms.ToTensor(),]
        ),
    )
testset = datasets.MNIST(
        mode="test", download=True,
        transform=transforms.Compose(
            [transforms.ToTensor()]
        ),
    )
trainset, validset = random_split(trainset, [opt.N_train, opt.N_valid])

traindataloader = DataLoader(
    trainset, batch_size=opt.batchsize, shuffle=True,
)
validdataloader = DataLoader(
    validset, batch_size=opt.N_valid, shuffle=True,
)
testdataloader = DataLoader(
    testset, batch_size=opt.N_test, shuffle=True,
)

valid_imgs, _ = next(iter(validdataloader))
test_imgs, _ = next(iter(testdataloader))

paddle.save(trainset, "./data/train")
paddle.save(valid_imgs, "./data/valid")
paddle.save(test_imgs, "./data/test")

logger.info('finish create datasets!')