import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=100, help="size of the batches")
    parser.add_argument("--gen_lr", type=float, default=0.0002, help="adam: generator learning rate")
    parser.add_argument("--reg_lr", type=float, default=0.0001, help="adam: reconstruction learning rate")
    parser.add_argument("--load", type=bool, default=False, help="load model or not")
    parser.add_argument("--N", type=bool, default=1200, help="Number of neurons")
    # data
    parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="Number of image channels")
    parser.add_argument("--latent_dim", type=int, default=8, help="dimensionality of the latent code")
    parser.add_argument("--std", type=float, default=5, help="std prior")
    parser.add_argument("--N_train", type=int, default=59000, help="Number of training data samples")
    parser.add_argument("--N_valid", type=int, default=1000, help="Number of valid data samples")
    parser.add_argument("--N_test", type=int, default=10000, help="Number of test data samples")
    parser.add_argument("--N_gen", type=int, default=10000, help="Number of generated data samples")
    parser.add_argument('--batchsize', type=int, default=100, help='input batch size for training (default: 100)')
    # test
    parser.add_argument("--load_epoch", type=int, default=199, help="the load model id")
    parser.add_argument('--sigma', type=float, default=None, help = "Window width")

    args = parser.parse_args()
    return args

