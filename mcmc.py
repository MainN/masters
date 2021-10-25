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
    def __init__(self, beta,gamma, warmup,yes,*args, **kwargs):
        self.beta = beta
        self.gamma = gamma
        self.model = SIR.SIR(1000,1,100,beta=self.beta,gamma=self.gamma,dt=4)
        self.x = [self.beta, self.gamma]
        self.yes = yes
    def log_likelihood(self,x,y):
        return tfp.distributions.Poisson(x).log_prob(y)
    def sum_log_likelihood(self,xes,yes):
       
        if not len(xes)==len(yes):
           raise Exception("Diff sizes")

        return np.sum( [self.log_likelihood (xes[t], yes[t]) for t in range (0, len(xes) ) ] )
    def proposal(self,x):        
        return [ tfp.distributions.HalfNormal (mu,0.1).sample().numpy() for mu in x]
    def run(self,iter):
        self.result = []
        for i in range(0,iter):
            x_new = self.proposal(self.x)
            self.model = SIR.SIR(1000,1,100,beta = self.x[0],gamma = self.x[1],dt=4)
            data_old = self.model.get_results()[:10]
            
            self.model = SIR.SIR(1000,1,100,beta=x_new[0],gamma=x_new[1],dt=4)
            
            data_new = self.model.get_results()[:10]
            
            log_lik_x = self.sum_log_likelihood(data_old,self.yes)
            log_lik_x_new = self.sum_log_likelihood(data_new,self.yes)
            
            if log_lik_x_new > log_lik_x:
                self.x = x_new
                self.result.append(x_new)
                print(log_lik_x,log_lik_x_new)