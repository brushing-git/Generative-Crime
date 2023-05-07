import numpy as np
from numpy.typing import ArrayLike
from . import utils as ut
from scipy.stats import multivariate_normal as norm

class Distribution:
    """
    Distribution class with the following properties:
    k: int : number of distributions
    rvs: int : the number of random variables
    dist: scipy.stats._continuous_distns :  the distributional form, e.g. multivariate Normal
    pi: (k) np.array floats : the priors over the distributions, initially uniform
    mus: (rvs,k) np.array floats : the means of the k distributions
    sigs: (rvs,rvs,k) np.array floats : the covariances of the k distributions
    """
    def __init__(self, k: int, mus: ArrayLike, sigs: ArrayLike) -> None:
        """
        Args:
        k : int : number of distributions
        mus : (rvs, k) np.array : the mus gotten from initializing
        sigs : (rvs, rvs, k) np.array : the covariances from initializing
        Output:
        None
        """
        self.k = k
        self.rvs = mus.shape[0]
        self.dist = norm
        self.pi = np.zeros(k)
        self.pi[:] = 1 / k

        # Set the initial means
        self.mus = mus
        
        # Set the initial covariances
        self.sigs = sigs
    
    def __str__(self):
        return "A mixture Gaussian Model"

    def e_step(self, x_i: ArrayLike) -> np.ndarray:
        """
        Computes the responsibility (ric) for each k distribution.

        ric = posterior predicted probability for each sample from priors

        Args: 
        x_i : (rvs) np.array floats : the rvs length sample
        Output: 
        ric : (k) np.array floats : the responsibility for each k distribution 
        """
        ric = np.ones(self.k)
        # cond_pdf is a (rvs,k) np.array of the probabilities
        cond_pdf = np.zeros(self.k)
        for i in range(self.k):
            cond_pdf[i] = self.dist.pdf(x_i, mean=self.mus[:,i], cov=self.sigs[:,:,i])
        
        # the denominator is the same for all normals so I compute it here
        denom = np.sum(self.pi*cond_pdf)
        for i in range(self.k):
            ric[i] = (self.pi[i]*cond_pdf[i]) / denom

        return ric
    
    def m_step(self, x: ArrayLike, rics: ArrayLike) -> None:
        """
        Computes the updated means and covariances.

        Args: 
        x : (rvs, n) np.array floats : n samples of rvs random variables
        rics : (n, k) np.array floats : ric of n samples from k distributions
        Output:  None : Updates self.mus and self.sigs
        """

        # Compute rk
        rk = np.zeros(self.k)
        for i in range(self.k):
            rk[i] = np.sum(rics[:,i])

        # Compute the means for each rvs (hence why we use the axis on the sum across n samples)
        for i in range(self.k):
            self.mus[:,i] = np.sum(x*rics[:,i], axis=1) / rk[i]
        
        # Compute the covariances
        for i in range(self.k):
            xmu = x.T - self.mus[:,i]
            sum = np.dot(rics[:,i]*xmu.T, xmu)
            self.sigs[:,:,i] = sum / rk[i]
        
        # Compute the priors
        for i, pi in enumerate(self.pi):
            pi = rk[i] / rics.shape[0]

    def pdf(self, x_i: ArrayLike) -> float:
        """
        Computes the probability density function for a specific sample.

        The formula is just weighted sum of the priors by their conditional density estimations, Sum pi * norm(x_i | mu, sig)

        Args:
        x_i : (rvs) np.array floats : 1 sample of rvs random variables
        Output: float : probability estimate from the model
        """

        prob = 0
        for i, pi in enumerate(self.pi):
            prob += pi * self.dist.pdf(x_i, mean=self.mus[:,i], cov=self.sigs[:,:,i])
        
        return prob


def train_model(model: Distribution, x_bar: ArrayLike, eps=0.001, max_epochs=100) -> None:
    """
    Training algorithm for the Gaussian Mixture Model.

    This works by training for a maximum number of epochs and stops either of the epochs stop or the NLL stops decreasing.

    Args:
    model : Distribution : a Distribution class model
    x_bar : (rvs, n) : n samples of rvs random variables
    eps : float : the maximum allowed change in difference to halt the training
    max_epochs : int : the maximum number of training epochs, initially set to 100,000
    Output: None : the model is updated during training
    """

    nll = 1.5 * ut.compute_nll(x_bar, model)
    nll_nex = ut.compute_nll(x_bar, model)
    epochs = 0

    while (abs(nll - nll_nex) / abs(nll)) > eps and epochs < max_epochs:
        # Compute nll to compare progress
        nll = ut.compute_nll(x_bar, model)

        # E Step
        rics = []
        for i in range(x_bar.shape[1]):
            ric = model.e_step(x_bar[:,i])
            rics.append(ric)
        
        rics = np.asarray(rics)

        # M Step
        model.m_step(x_bar, rics)
        
        # Compute new nll
        nll_nex = ut.compute_nll(x_bar, model)

        # Increment count
        epochs += 1

        # Print current loss and epoch
        print('Epoch ' + str(epochs) + ' has a current loss of ' + str(nll_nex) + '.')
        print('The estimated difference for the next step is ' + str(abs(nll - nll_nex)))