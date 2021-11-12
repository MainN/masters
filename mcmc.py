import networkx as nx
import numpy as np
from numpy import random
import SIR_model as SIR


class MCMC():
    def __init__(self, size, percent, t, pseudodata, beta_0, gamma_0, dt):
        # set initial values
        self.beta = beta_0
        self.gamma = gamma_0
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
        return k*np.log(l) - l

    def sum_log_likelihood(self, xes):
        if len(xes) != len(self.pseudodata):
            raise Exception(f"Diff sizes: {len(xes)}, {len(self.pseudodata)}")

        # likelihood can be parralelized with numpy arrays
        return self.log_likelihood(k=self.pseudodata, l=np.array(xes)).sum()

    def proposal(self, x):
        return [abs(random.normal(mu, 0.001)) for mu in x]

    def run(self,iter):
        self.result = []         # samples would go here
        self.extra_result = []   # additional information will go here

        # epidemic trajectory and likelihood on the current iteration
        self.model = SIR.SIR(beta=self.x[0], gamma=self.x[1], **self.model_settings)
        cur_tr = self.model.get_results_2()
        cur_likelihood = self.sum_log_likelihood(cur_tr)

        for i in range(iter):
            # sample new parameters
            x_new = self.proposal(self.x)

            # calculate new likelihood
            self.model = SIR.SIR(beta=x_new[0], gamma=x_new[1], **self.model_settings)
            new_tr = self.model.get_results_2()
            new_likelihood = self.sum_log_likelihood(new_tr)

            # new parameter is accepted if
            # U(0, 1) < new_likelihood / current_likelihood
            # applying log transformation to both sides, we age getting
            # -Exp(1) < log(new_likelihood) - log(current_likelihood)
            # where Exp(1) is an exponential distribution
            accept = -random.exponential(1) < new_likelihood - cur_likelihood
            if accept:
                self.x = x_new
                cur_tr = new_tr
                cur_likelihood = new_likelihood

            # record the iteration
            self.result.append(self.x)
            self.extra_result.append([cur_likelihood, accept])


if __name__ == '__main__':
    mcmc = MCMC(size=100, percent=1, t=16, pseudodata=[1, 2, 4, 8, 15, 25, 35, 40, 40, 30, 20, 10, 5, 3, 2, 1], beta_0=1, gamma_0=1, dt=1)
    mcmc.run(iter=1000)
    print(mcmc.x)
