import pickle
import argparse

import paddle
from paddle.io import random_split
from paddle.io import DataLoader
import paddle.vision.transforms as transforms

from network import Encoder, Decoder, Discriminator
from utils.parzen_ll import *
from utils.log import get_logger
from config import args_parser

def load_data():
    trainset = paddle.load("./data/train")
    traindataloader = DataLoader(
        trainset, batch_size=opt.batchsize, shuffle=True,
    )
    valid_imgs = paddle.load("./data/valid")
    test_imgs = paddle.load("./data/test")

    return traindataloader,valid_imgs,test_imgs

if __name__ == "__main__":
    # Training settings
    opt = args_parser()

    logger = get_logger('./logs/eval.log')
    logger.info(opt)

    checkpoint = paddle.load("./model/model" + str(opt.load_epoch) + ".pkl")
    encoder = Encoder()
    decoder = Decoder()
    encoder_dict = checkpoint['encoder']
    decoder_dict = checkpoint['decoder']
    encoder.set_state_dict(encoder_dict)
    decoder.set_state_dict(decoder_dict)
    encoder.eval()
    decoder.eval()
    logger.info("model/model%d.pkl loaded!" % opt.load_epoch)

    # preprocessing 
    z = paddle.normal(0,opt.std,(opt.N_gen, opt.latent_dim))
    traindataloader,valid_imgs,test_imgs = load_data()

    gen_imgs = decoder(z)
    train_imgs, _ = next(iter(traindataloader))
    
    gen_imgs = paddle.reshape(gen_imgs, (opt.N_gen, -1))
    train_imgs = paddle.reshape(train_imgs, (opt.batchsize, -1))
    valid_imgs = paddle.reshape(valid_imgs, (opt.N_valid, -1))
    test_imgs = paddle.reshape(test_imgs, (opt.N_test, -1))

    gen = np.asarray(gen_imgs.detach().cpu())
    train = np.asarray(train_imgs.detach().cpu())
    test = np.asarray(test_imgs.detach().cpu())
    valid = np.asarray(valid_imgs.detach().cpu())
    
    # cross validate sigma
    if opt.sigma is None:
        sigma_range = np.logspace(start = -1, stop = -0.3, num=20)
        sigma = cross_validate_sigma(
            gen, valid, sigma_range, batch_size = opt.batchsize, logger = logger
        )
        opt.sigma = sigma
    else: 
        sigma = float(opt.sigma)
    logger.info("Using Sigma: {}".format(sigma))

    # fit and evaulate
    # gen_imgs
    parzen = parzen_estimation(gen, sigma)
    ll = get_nll(test, parzen, batch_size = opt.batchsize)
    se = ll.std() / np.sqrt(test.shape[0])
    logger.info("Log-Likelihood of test set = {}, se: {}".format(ll.mean(), se))
    ll = get_nll(valid, parzen, batch_size = opt.batchsize)
    se = ll.std() / np.sqrt(valid.shape[0])
    logger.info("Log-Likelihood of valid set = {}, se: {}".format(ll.mean(), se))

    logger.info("finish evaluation")

