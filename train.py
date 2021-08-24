import argparse
import os
import numpy as np
import math
import itertools
import pickle
import pandas as pd

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.vision.transforms as transforms
from paddle.io import random_split
from paddle.io import DataLoader
from paddle.vision import datasets
from paddle.vision.datasets import MNIST

from network import Encoder, Decoder, Discriminator
from utils.paddle_save_image import save_image
from utils.parzen_ll import *
from utils.log import get_logger

# paddle.utils.run_check()

def sample_image(n_row, epoch):
    """Saves a grid of generated digits"""
    # Sample noise
    z = paddle.normal(0, opt.std, (n_row ** 2, opt.latent_dim))
    gen_imgs = decoder(z)
    # gen_imgs = paddle.to_tensor(gen_imgs)
    save_image(gen_imgs, "images/epoch%3d.png" % epoch, nrow=n_row, normalize=True)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.__class__.weight_attr = nn.initializer.Normal(0.0, 0.01)
    elif classname.find("BatchNorm2d") != -1:
        m.__class__.weight_attr = nn.initializer.Normal(1.0, 0.02)
        m.__class__.weight_attr = nn.initializer.Constant(0.0)

def pd_one_epoch_to_csv(export_data, epoch, D_loss, G_loss, PATH, recon_loss=None):
    export_data_line = np.zeros(3)
    export_data_line[0] = D_loss.item()
    export_data_line[1] = G_loss.item()
    export_data_line[2] = recon_loss.item()
    export_data.append(export_data_line.reshape(-1,))
    data = np.array(export_data)
    data = pd.DataFrame(data=data) 
    data.to_csv(PATH,index = True)
    return export_data

if __name__ == "__main__" :
    loss = []
    device_id = 0
    os.makedirs("images", exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=100, help="size of the batches")
    parser.add_argument("--gen_lr", type=float, default=0.0002, help="adam: generator learning rate")
    parser.add_argument("--reg_lr", type=float, default=0.0001, help="adam: reconstruction learning rate")
    parser.add_argument("--latent_dim", type=int, default=8, help="dimensionality of the latent code")
    parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--std", type=float, default=5, help="std prior")
    parser.add_argument("--load", type=bool, default=False, help="load model or not")
    opt = parser.parse_args()
    # log 输出
    logger = get_logger('./logs/train.log')
    logger.info(opt)

    img_shape = (opt.channels, opt.img_size, opt.img_size)

    # Configure data loader
    trainset = paddle.load("./data/train")
    traindataloader = DataLoader(
        trainset, batch_size=100, shuffle=True,
    )

    # Initialize generator and discriminator
    encoder = Encoder()
    decoder = Decoder()
    discriminator = Discriminator()

    encoder.apply(weights_init_normal)
    decoder.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
    # Optimizers
    # Set optimizators
    P_decoder = paddle.optimizer.Adam(
        parameters = decoder.parameters(), 
        learning_rate=opt.gen_lr)
    Q_encoder = paddle.optimizer.Adam(
        parameters = encoder.parameters(), 
        learning_rate=opt.gen_lr)
    Q_generator = paddle.optimizer.Adam(
        parameters = encoder.parameters(), 
        learning_rate=opt.reg_lr)
    D_gauss_solver = paddle.optimizer.Adam(
        parameters = discriminator.parameters(), 
        learning_rate=opt.reg_lr)

    if opt.load == True:
        checkpoint = paddle.load("./model/model299.pkl")
        encoder.set_state_dict(checkpoint['encoder'])
        decoder.set_state_dict(checkpoint['decoder'])
        discriminator.set_state_dict(checkpoint['discriminator'])
        P_decoder.set_state_dict(checkpoint['P_decoder'])
        Q_encoder.set_state_dict(checkpoint['Q_encoder'])
        Q_generator.set_state_dict(checkpoint['Q_generator'])
        D_gauss_solver.set_state_dict(checkpoint['D_gauss_solver'])

    TINY = 1e-15
    # ----------
    #  Training
    # ----------

    for epoch in range(opt.n_epochs):
        for i, (imgs, _) in enumerate(traindataloader):
            encoder.train()
            decoder.train()
            discriminator.train()

            # Adversarial ground truths
            valid = paddle.to_tensor(np.ones((imgs.shape[0], 1)), dtype='float32', stop_gradient = True)
            fake = paddle.to_tensor(np.zeros((imgs.shape[0], 1)), dtype='float32', stop_gradient = True)
            # Configure input
            real_imgs = imgs

            #######################
            # Reconstruction phase
            #######################
            z_sample = encoder(real_imgs)
            X_sample = decoder(z_sample)
            recon_loss = F.binary_cross_entropy(X_sample + TINY, real_imgs + TINY)

            recon_loss.backward()
            P_decoder.step()
            Q_encoder.step()

            P_decoder.clear_grad()
            Q_encoder.clear_grad()
            Q_generator.clear_grad()
            D_gauss_solver.clear_grad()
            #######################
            # Regularization phase
            #######################
            # Discriminator
            encoder.eval()
            z_real_gauss = paddle.normal(0, opt.std, (imgs.shape[0], opt.latent_dim))
            z_fake_gauss = encoder(real_imgs)
            z_fake_gauss.stop_gradient = True # 阻止梯度回传
            D_real_gauss = discriminator(z_real_gauss)
            D_fake_gauss = discriminator(z_fake_gauss)

            D_loss = -paddle.mean(paddle.log(D_real_gauss + TINY) + paddle.log(1 - D_fake_gauss + TINY))
            D_loss.backward()
            D_gauss_solver.step()

            P_decoder.clear_grad()
            Q_encoder.clear_grad()
            Q_generator.clear_grad()
            D_gauss_solver.clear_grad()
            # Generator
            encoder.train()
            z_fake_gauss = encoder(real_imgs)
            D_fake_gauss = discriminator(z_fake_gauss)
            G_loss = -paddle.mean(paddle.log(D_fake_gauss + TINY))

            G_loss.backward()
            Q_generator.step()

            P_decoder.clear_grad()
            Q_encoder.clear_grad()
            Q_generator.clear_grad()
            D_gauss_solver.clear_grad()

        encoder.eval()
        decoder.eval()
        discriminator.eval()

        logger.info(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [recon loss: %f]"
                % (epoch, opt.n_epochs, i, len(traindataloader), D_loss.item(), G_loss.item(), recon_loss.item())
            )
        loss = pd_one_epoch_to_csv(loss, epoch, D_loss, G_loss, "./logs/loss.csv", recon_loss = recon_loss)
        if epoch % 10 == 0:
            sample_image(n_row=10, epoch=epoch)
            logger.info("images%d saved in ./images/images%d.png" % (epoch, epoch))
        # 计算结果
        if (epoch+1) % 50 == 0:
            logger.info("model%d saved in ./model/model%d.pkl" % (epoch, epoch))
            paddle.save({
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'discriminator':discriminator.state_dict(),
                'P_decoder': P_decoder.state_dict(),
                'Q_encoder': Q_encoder.state_dict(),
                'Q_generator': Q_generator.state_dict(),
                'D_gauss_solver': D_gauss_solver.state_dict(),
            }, 
            str("./model/model" + str(epoch) + ".pkl") )

    logger.info("finish training")
            
            
    
            
    