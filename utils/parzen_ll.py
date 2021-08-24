import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import gc
from utils.log import get_logger

def get_nll(x, parzen, batch_size=10):
    """
    Credit: Yann N. Dauphin
    """
    inds = np.arange(x.shape[0])
    n_batches = int(np.ceil(float(len(inds)) / batch_size))
    nlls = []
    for i in range(n_batches):
        nll = parzen(x[inds[i::n_batches]])
        nlls.extend(nll)
    return np.array(nlls)


def cross_validate_sigma(samples, data, sigmas, batch_size, logger = None):
    lls = []
    for sigma in sigmas:
        parzen = parzen_estimation(samples, sigma, mode='gauss')
        tmp = get_nll(data, parzen, batch_size = batch_size)
        tmp_mean = np.asarray(tmp).mean()
        lls.append(tmp_mean)
        if logger == None:
            print("sigma = %.5f, nll = %.5f" % (sigma, tmp_mean))
        else:
            logger.info("sigma = %.5f, nll = %.5f" % (sigma, tmp_mean))
        del parzen
        gc.collect()
    ind = np.argmax(lls)
    return sigmas[ind]


def parzen_estimation(mu, sigma, mode='gauss'):
    """
    Implementation of a parzen-window estimation
    Keyword arguments:
        x: A "nxd"-dimentional numpy array, which each sample is
                  stored in a separate row (=training example)
        mu: point x for density estimation, "dx1"-dimensional numpy array
        sigma: window width
    Return the density estimate p(x)
    """

    def log_mean_exp(a):
        max_ = a.max(axis=1)
        return max_ + np.log(np.exp(a - np.expand_dims(max_, axis=1)).mean(1))

    def gaussian_window(x, mu, sigma):
        a = (np.expand_dims(x, axis=1) - np.expand_dims(mu, axis=0)) / sigma
        b = np.sum(- 0.5 * (a ** 2), axis=-1)
        E = log_mean_exp(b)
        Z = mu.shape[1] * np.log(sigma * np.sqrt(np.pi * 2))
        return (E - Z)

    def hypercube_kernel(x, mu, h):
        n, d = mu.shape
        a = (np.expand_dims(x, axis=1) - np.expand_dims(mu, axis=0)) / h
        b = np.all(np.less(np.abs(a), 1/2), axis=-1)
        kn = np.sum(b.astype(int), axis=-1)
        return kn / (n * h**d)

    if mode == 'gauss':
        return lambda x: gaussian_window(x, mu, sigma)
    elif mode == 'hypercube':
        return lambda x: hypercube_kernel(x, mu, h=sigma)

# Pdf estimation
def pdf_multivaraible_gauss(x, mu, cov):
    part1 = 1 / ((2 * np.pi) ** (len(mu)/2) * (np.linalg.det(cov)**(1/2)))
    part2 = (-1/2) * (x-mu).T.dot(np.linalg.inv(cov)).dot((x-mu))
    return float(part1 * np.exp(part2))

# if __name__ == "__main__":
#     # Make data
#     np.random.seed(2017)
#     N = 1024    # features map
#     mu_vec = np.zeros(N)
#     cov_mat = np.identity(N)
#     # x_gen_samples = np.random.multivariate_normal(mu_vec, cov_mat, 10000)
#     # x_valid = np.random.multivariate_normal(mu_vec, cov_mat, 1000)
#     # x_test = np.random.multivariate_normal(mu_vec, cov_mat, 1000)
#     x_gen_samples = np.random.randint(0,2, (10000, N))
#     x_valid = np.random.randint(0,2, (1000, N))
#     x_test = np.random.randint(0,2, (10000, N))
#     sigmas_samples = np.logspace(-1, 1, 10)

#     sigma = cross_validate_sigma(x_gen_samples, x_valid, sigmas_samples, batch_size=100)
#     print(sigma)
#     fig = plt.figure(figsize=(6, 6))
#     ax = fig.gca()
#     ax.plot(sigmas_samples, nll,  '-*r')
#     ax.set(xlabel='log likelihood', ylabel='\sigma value', title='NLL method ')
#     plt.show()

#     pg = parzen_estimation(x_gen_samples, sigma, mode='gauss')
#     print(pg(np.zeros((1,N))))

#     ll = get_nll(x_test, pg, batch_size = 100)
#     se = ll.std() / np.sqrt(x_test.shape[0])
#     print("Log-Likelihood of test set = {}, se: {}".format(ll.mean(), se))

    