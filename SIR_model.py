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
class SIR():
    def __init__(self,size,percent,t,*args, beta, gamma, dt,**kwargs):
    #инициализируем изначальное состояние модели
        self.I = int(size/100*percent)
        self.S = size-self.I
        self.R = 0
        self.N = size
        self.beta = beta
        self.gamma = gamma
        self.result_S = []
        self.result_I = []
        self.result_R = []
        self.dt = dt
        self.t = t
        self.steps = round(t/dt)
        
    def calc(self):
        #вычисление 
        for t in range(0,self.steps):
            #считаем приращения
            ds_dt=self.dS_dt()
            di_dt=self.dI_dt()
            dr_dt=self.dR_dt()
            
            #добавляем в результирующие списки
            self.result_S.append(ds_dt )
            self.result_I.append(di_dt )
            self.result_R.append(dr_dt )
            
            #пересчитываем количество людей в группах
            self.S += ds_dt
            self.I += di_dt
            self.R += dr_dt
          
        #выводим количество индивидуумов в каждой группе
        #print(self.S,self.I,self.R)
        
    def dS_dt(self):
        return (-self.beta * self.S * self.I / self.N) * self.dt
    def dR_dt(self):
        return self.gamma * self.I*self.dt
    def dI_dt(self):
        return (self.beta * self.S * self.I / self.N - self.gamma * self.I)*self.dt
        
    def viz(self):
        #Собираем в табличку полученные списки прирощений
        table=list(zip(self.result_S,self.result_I,self.result_R))
        #Создаем табличку pandas
        df = pd.DataFrame(table,
               columns =['S', 'I','R'])
        #возвращаем визуализацию
        return (sns.lineplot(data=df))
    def set_params(self,params):
        self.beta=params[0]
        self.gamma=params[1]
        
        #print(self.beta,self.gamma)
    def get_results(self):
        
        self.calc()
        #print((self.beta,self.gamma),self.result_S)
        return [abs(x) for x in self.result_S]
    def get_R0(self):
        return self.beta/self.gamma