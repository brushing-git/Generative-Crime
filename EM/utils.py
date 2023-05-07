import numpy as np
from numpy.typing import ArrayLike

def compute_nll(x: ArrayLike, model) -> float:
    """
    Computes the negative log-likelihood from a probability distribution.
    The NLL is - Sum log pdf for all samples.

    Args:
    x: (rvs,n) np.array : numpy array of (rvs, n) floats where rvs is the number of random variables
    model: Distribution : a probability distribution that has the .pdf method

    Output:
    nll: float : the negative log likelihood
    """
    n = x.shape[1]
    ll = np.zeros(n)
    for i, x_i in enumerate(np.rollaxis(x, 1)):
        ll[i] = model.pdf(x_i)

    ll = np.log(ll)

    nll = - np.sum(ll)
    return nll

def init_data(x: ArrayLike, k: int) -> np.ndarray:
    """
    Returns the initial means and covariances from a data set.

    Args:
    x: (rvs, n) np.array : numpy array of (rvs, n) floats where rvs is the number of random variables
    k: int : the number of clusters

    Output:
    means: (rvs, k) np.array : numpy array of (rvs, k) floats that are means for each k clusters
    covs: (rvs, rvs, k) np.array : numpy array of (rvs, rvs, k) floats that are the covariances for each k cluster
    """

    rvs = x.shape[0]
    means = np.zeros((rvs,k))
    for i in range(k):
        indx = np.random.randint(low=0, high=x.shape[1])
        means[:,i] = x[:,indx]
    
    covs = np.zeros((rvs,rvs,k))
    for i in range(k):
        covs[:,:,i] = np.cov(x)
    
    return means, covs
