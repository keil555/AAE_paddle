import pickle
import argparse

import paddle
from paddle.io import random_split
from paddle.io import DataLoader
import paddle.vision.transforms as transforms

from network import Encoder, Decoder, Discriminator
from utils.parzen_ll import *
from utils.log import get_logger

def load_data():
    trainset = paddle.load("./data/train")
    traindataloader = DataLoader(
        trainset, batch_size=100, shuffle=True,
    )
    valid_imgs = paddle.load("./data/valid")
    test_imgs = paddle.load("./data/test")

    return traindataloader,valid_imgs,test_imgs

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='MNIST')

    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--sigma', type=float, default=None, help = "Window width")
    parser.add_argument("--std", type=float, default=5, help="std prior")
    parser.add_argument("--latent_dim", type=int, default=8, help="dimensionality of the latent code")
    parser.add_argument("--epoch", type=int, default=199, help="the load model id")
    opt = parser.parse_args()
    logger = get_logger('./logs/eval.log')
    logger.info(opt)

    checkpoint = paddle.load("./model/model" + str(opt.epoch) + ".pkl")
    encoder = Encoder()
    decoder = Decoder()
    encoder_dict = checkpoint['encoder']
    decoder_dict = checkpoint['decoder']
    encoder.set_state_dict(encoder_dict)
    decoder.set_state_dict(decoder_dict)
    encoder.eval()
    decoder.eval()
    logger.info("model/model%d.pkl loaded!" % opt.epoch)

    # preprocessing 
    z = paddle.normal(0,opt.std,(10000, opt.latent_dim))
    traindataloader,valid_imgs,test_imgs = load_data()

    gen_imgs = decoder(z)
    train_imgs, _ = next(iter(traindataloader))
    
    gen_imgs = paddle.reshape(gen_imgs, (10000, -1))
    train_imgs = paddle.reshape(train_imgs, (100, -1))
    valid_imgs = paddle.reshape(valid_imgs, (1000, -1))
    test_imgs = paddle.reshape(test_imgs, (10000, -1))

    gen = np.asarray(gen_imgs.detach().cpu())
    train = np.asarray(train_imgs.detach().cpu())
    test = np.asarray(test_imgs.detach().cpu())
    valid = np.asarray(valid_imgs.detach().cpu())
    
    # cross validate sigma
    if opt.sigma is None:
        sigma_range = np.logspace(start = -1, stop = -0.3, num=20)
        sigma = cross_validate_sigma(gen, valid, sigma_range, batch_size = 100, logger = logger)
        opt.sigma = sigma
    else: 
        sigma = float(opt.sigma)
    logger.info("Using Sigma: {}".format(sigma))

    # fit and evaulate
    # gen_imgs
    parzen = parzen_estimation(gen, sigma)
    ll = get_nll(test, parzen, batch_size = 100)
    se = ll.std() / np.sqrt(test.shape[0])
    logger.info("Log-Likelihood of test set = {}, se: {}".format(ll.mean(), se))
    ll = get_nll(valid, parzen, batch_size = 100)
    se = ll.std() / np.sqrt(valid.shape[0])
    logger.info("Log-Likelihood of valid set = {}, se: {}".format(ll.mean(), se))

    logger.info("finish evaluation")

