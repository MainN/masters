import networkx as nx
import numpy as np
from numpy import random
import tensorflow as tf
import tensorflow_probability as tfp
import SIR_model as SIR


class MCMC():
    def __init__(self, size, percent, t, warmup, pseudodata, beta_0, gamma_0, dt):
        # set initial values
        self.beta = beta
        self.gamma = gamma
        self.x = [self.beta, self.gamma]

        # set up the model
        self.model_settings = dict(size=size, percent=percent, dt=dt, t=t)
        self.model = SIR.SIR(beta=self.beta, gamma=self.gamma, **self.model_settings)

        self.pseudodata = np.array(pseudodata)

    def log_likelihood(self, k, l):
        # possion distribution is defined as
        # poisson(k | λ) = λ**k / k! * exp(-λ)
        # factors independet from λ are irrelevant for MCMC, remove them:
        # poisson(k | λ) ∝ λ**k * exp(-λ)
        # we save time by not computing the factorial
        return x*np.log(l) - l

    def sum_log_likelihood(self, xes):
        if len(xes) != len(self.pseudodata):
            raise Exception("Diff sizes")

        # likelihood can be parralelized with numpy arrays
        return self.log_likelihood(k=self.pseudodata, l=np.array(xes)).sum()

    def proposal(self,x):
        return [abs(random.normal(mu, 0.001)) for mu in x]

    def run(self,iter):
        self.result = []
        # epidemic trajectory and likelihood on the current iteration
        self.model = SIR.SIR(beta=self.x[0], gamma=self.x[1], **self.model_settings)
        cur_tr = self.model.get_results(self.pseudodata)
        cur_likelihood = self.sum_log_likelihood(cur_tr)

        for i in range(iter):
            # sample new parameters
            x_new = self.proposal(self.x)

            # calculate new likelihood
            self.model = SIR.SIR(beta=x_new[0], gamma=x_new[1], **self.model_settings)
            new_tr = self.model.get_results(self.pseudodata)
            new_likelihood = self.sum_log_likelihood(new_tr)

            # new parameter is accepted if
            # U(0, 1) < new_likelihood / current_likelihood
            # applying log transformation to both sides, we age getting
            # -Exp(1) < log(new_likelihood) - log(current_likelihood)
            # where Exp(1) is an exponential distribution
            if -random.exponential(1) < new_likelihood - cur_likelihood:
                self.x = x_new
                cur_tr = new_tr
                cur_likelihood = new_likelihood

            # record the iteration
            self.result.append(self.x)
