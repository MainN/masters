# -*- coding: utf-8 -*-
"""graph.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1O4maIOx-pZT2PmR5kl84M-LpCtq34x7T
"""

try:
    from google.colab import drive
    drive.mount('/content/gdrive')
    import sys
    sys.path.append('/content/gdrive/MyDrive')
except ImportError:
    pass


import random

import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
import matplotlib
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

try:
    from matplotlib.animation import PillowWriter
except ImportError:
    pass

sns.set()


class EpiProcess():
    def __init__(self, *args, **kwargs):
        #вызываем стартовые инциализаторы
        self.paramertrs_init(*args, **kwargs)
        self.graph_init()
        self.random_start_sample()

        if self.viz:
            self.viz_init(viz)

    def paramertrs_init(self, size, percent, viz=False):
        #инициализируем параметры процесса
        self.size = size
        self.percent = percent
        self.cummulitive_sum_I = 0
        self.iterations = []
        self.viz=viz

    def graph_init(self):
        #строим configuration_model
        sequence = nx.random_powerlaw_tree_sequence(self.size, tries=5000000)
        self.G = nx.configuration_model(sequence)

    def viz_init(self,viz):
        #инциализируем необходимое для визуализации распростронения
        self.fig = None
        self.axes = None
        self.pos = nx.spring_layout(self.G)
        self.gif=[]
        self.colors = sns.color_palette("hls", 8)

    def random_start_sample(self):
        #Инициализация множеств индивидуумов, изначальной выборки
        self.start_sample_size = int(self.size/100 * self.percent)
        self.S = set(self.G)
        self.I = set()
        self.R = set()
        self.tmp = set()
        self.result = []

        #Заражаем изначальный процент
        while len(self.I)<self.start_sample_size:
            self.infect(random.randint(0, self.size))

    def infect(self, x):
        #Заражаем конкретного индивиидума путём перемещения его из множества
        if x in self.S:
            self.S.discard(x)
            self.I.add(x)

    def recover(self, x):
        #Конкретный индивидуум переболел перемещаем его в множество переболевших
        self.I.discard(x)
        self.R.add(x)

    def infect_neigh(self, x):
        #метод заражения соседей
        for neigh in self.G.neighbors(x):
            self.infect(neigh)

    def iterartion(self):
        #метод итерации в графе
        #создаем копию больных что бы ничего не испортить
        #для больных заражаем соседей а сам больной восстанавливается
        for x in self.I.copy():
            self.infect_neigh(x)
            self.recover(x)

    def run(self):
        if self.viz == True:
            #если включена визуализация
            while len(self.I) != 0:
                #добавляем текущую итерацию в список картинок
                self.gif.append(self.vis_spread_info())

                #добавляем в результирующий список больных на текущий момент
                self.result.append(len(self.I))

                #проводим следующую итерацию
                self.iterartion()

            #повторяем для последней итерации
            self.result.append(len(self.I))
            self.gif.append(self.vis_spread_info())
        else:
            #если визуализация выключена
            while len(self.I) != 0:
                 #добавляем в результирующий список больных на текущий момент
                self.result.append(len(self.I))

                #проводим следующую итерацию
                self.iterartion()

            #повторяем для последней итерации
            self.result.append(len(self.I))
    def viz_run(self):
        #метод визуализации распростронения по количеству больных на момент времени
        #получаем из списка результа данные в нужном формате
        y = pd.Series(self.result, name="count")

        #конвертируем итерации в моменты врмени
        x = pd.Series(range(1,len(self.result)+2), name="iteration")

        return sns.lineplot(x,y)

    def plot_degree_dist(self):
        #лямда перебора количества соседей для каждой вершины
        degrees = [self.G.degree(n) for n in self.G.nodes()]

        sns.displot(degrees)

    def plot_degree_dist_neigh(self):
        #получаем график распредления среднего количества соседей среди сосдеей для каждой вершины графа
        sns.displot(nx.average_neighbor_degree(self.G))

    def vis_spread_info(self):
        #получаем актуальные цвета вершин на текущую итерацию
        colors = (list([self.colors[0]] * len(self.S)) +
                  list([self.colors[1]] * len(self.I)) +
                  list([self.colors[2]] * len(self.R)))

        #получаем актуальную принадлежность множествам вершин на текущую итерацию
        nodes = (list(self.S) +
                 list(self.I) +
                 list(self.R))

        return nodes,colors

    def update(self,num):
        #рисуем актуальный граф на итерацию Num
        nx.draw_networkx_nodes(self.G, pos=self.pos, nodelist=self.gif[num][0], node_color=self.gif[num][1])

    def viz_joint(self):
        #получаем данные
        neigh=nx.average_neighbor_degree(self.G)
        degrees = [self.G.degree(n) for n in self.G.nodes()]
        #конвертируем в пандас
        y = pd.Series(neigh, name="avg_neigh_degree")

        x = pd.Series(degrees, name="degrees")

        #рисуем
        sns.jointplot(x,y)


if __name__ == '__main__':
    ep = EpiProcess(size=1000, percent=1)
    ep.run()
    print(ep.result)
