import networkx as nx
import numpy as np
import random
import pandas as pd
import seaborn as sns
#import epi_process as ep
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from matplotlib.animation import FuncAnimation
sns.set()
import tensorflow as tf
import tensorflow_probability as tfp
import SIR_model as SIR


class MCMC():
    def __init__(self, size,percent,t, warmup, pseudodata, *args,beta,gamma, dt, **kwargs):
        self.beta = beta
        self.gamma = gamma
        self.size = size
        self.percent = percent
        self.t = t
        self.dt = dt
        self.model = SIR.SIR(self.size,self.percent,self.t,beta=self.beta,gamma=self.gamma,dt=self.dt)
        self.x = [self.beta, self.gamma]
        self.pseudodata = np.array(pseudodata)
        self.reject=[]

    def log_likelihood(self, k, l):
        # possion distribution is defined as
        # poisson(k | λ) = λ**k / k! * exp(-λ)
        # factors independet from λ are irrelevant for MCMC, remove them:
        # poisson(k | λ) ∝ λ**k * exp(-λ)
        # we save time by not computing the factorial
        return x*np.log(l) - l

    def sum_log_likelihood(self, xes):
        if not len(xes)==len(yes):
            raise Exception("Diff sizes")

        # likelihood can be parralelized with numpy arrays
        return self.log_likelihood(k=self.pseudodata, l=np.array(xes)).sum()

    def proposal(self,x):
        return [ abs(tfp.distributions.Normal(mu,0.001).sample().numpy()) for mu in x]

    def run(self,iter):
        self.result = []
        for i in range(0,iter):
            x_new = self.proposal(self.x)
            self.model = SIR.SIR(self.size,self.percent,self.t,beta = self.x[0],gamma = self.x[1],dt=self.dt)
            data_old = self.model.get_results(self.yes)

            self.model = SIR.SIR(self.size,self.percent,self.t,beta=x_new[0],gamma=x_new[1],dt=self.dt)

            data_new = self.model.get_results(self.yes)

            log_lik_x = self.sum_log_likelihood(data_old)
            log_lik_x_new = self.sum_log_likelihood(data_new)

            if log_lik_x_new > log_lik_x:
                self.x = x_new
                self.result.append(x_new)
                #print(log_lik_x,log_lik_x_new)
            else:
                alpha=tfp.distributions.Uniform(0.,1.).sample().numpy()
                #print( np.log(alpha),log_lik_x_new-log_lik_x)
                if  np.log(alpha )< ((log_lik_x_new-log_lik_x)):
                    self.x = x_new
                    self.result.append(x_new)
                else:
                    self.result.append(self.x)