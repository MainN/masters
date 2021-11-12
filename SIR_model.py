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
            self.result_S.append(ds_dt)
            self.result_I.append(di_dt)
            self.result_R.append(dr_dt)

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

    def set_params(self, params):
        self.beta = params[0]
        self.gamma = params[1]
        #print(self.beta,self.gamma)

    def get_results(self,xes):
        self.calc()
        self.result_S = self.normalize(xes)
        #print((self.beta,self.gamma),self.result_S)
        return [abs(x) for x in self.result_S]
    def get_results_2(self):
        self.calc()
        t = 0
        ipt = int(1/self.dt)  # iteration per time unit
        return [-sum(self.result_S[t:t+ipt]) for t in range(0, self.steps, ipt)]

    def get_R0(self):
        return self.beta/self.gamma

    def normalize(self,lst):
        pos=-1
        for x in range(0,len(self.result_S)):
            if self.result_S[x]>-1:
                pos=x
                break
        nrm_pos = len(lst)
        pairs = pos-nrm_pos
        self.res = self.result_S[:pos]
        while len(self.res)>2*nrm_pos:
            tmp_list=[]
            for x in range(0,int(pos/2)-1):
                tmp_list.append(self.res[x*2]+self.res[x*2+1])
            self.res=tmp_list
            pos=int(pos/2)
        tmp_list=[]
        for x in range(0,len(self.res)-nrm_pos):
            tmp_list.append(self.res[x*2]+self.res[x*2+1])

        return [*tmp_list,*self.res[(2*len(tmp_list)):]]


if __name__ == '__main__':
    sir = SIR(size=1000, percent=1, t=10, beta=2, gamma=1, dt=0.1)
    print(sir.get_results_2())
