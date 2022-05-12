import random as random
import matplotlib.pyplot as plt
import scipy.stats as sps
import numpy as np
from numpy import arange

sample = sps.norm(loc=1, scale=3).rvs(size=200)
print('Первые 10 значений выборки:\n', sample[:10])
print('Выборочное среденее: %.3f' % sample.mean())
print('Выборочная дисперсия: %.3f' % sample.var())
print('Плотность:\t\t', sps.norm(loc=1, scale=3).pdf([-1, 0, 1, 2, 3]))
print('Функция распределения:\t', sps.norm(loc=1, scale=3).cdf([-1, 0, 1, 2, 3]))

def show_pdf(pdf, xmin, xmax, ymax, grid_size, distr_name, **kwargs):
    """
    Рисует график плотности непрерывного распределения

    pdf - плотность
    xmin, xmax - границы графика по оси x
    ymax - граница графика по оси y
    grid_size - размер сетки, по которой рисуется график
    distr_name - название распределения
    kwargs - параметры плотности
    """

    grid = np.linspace(xmin, xmax, grid_size)
    plt.figure(figsize=(12, 5))
    plt.plot(grid, pdf(grid, **kwargs), lw=5)
    plt.grid(ls=':')
    plt.xlabel('Значение', fontsize=18)
    plt.ylabel('Плотность', fontsize=18)
    plt.xlim((xmin, xmax))
    plt.ylim((None, ymax))
    title = 'Плотность {}'.format(distr_name)
    plt.title(title.format(**kwargs), fontsize=20)
    plt.show()

show_pdf(
    pdf=sps.norm.pdf, xmin=-3, xmax=3, ymax=0.5, grid_size=100,
    distr_name=r'$N({loc}, {scale})$', loc=0, scale=1
)

sample = sps.norm.rvs(size=100000)  # выборка размера 1000
#аналитика
grid = np.linspace(-3, 3, 1000)  # сетка для построения графика
plt.figure(figsize=(16, 7))
plt.hist(sample, bins=30, density=True,
         alpha=0.6, label='Гистограмма выборки')
plt.plot(grid, sps.norm.pdf(grid), color='red',
         lw=5, label='Плотность случайной величины')
plt.title(r'Случайная величина $\xi \sim \mathcal{N}$(0, 1)', fontsize=20)
plt.legend(fontsize=14, loc=1)
plt.grid(ls=':')
plt.show()

X = 30 #Birthday
line=range(0, 65+X)
mychoise = []
mychoise2 = []
for i in range(0, 65+X):
    mychoise.append(random.choice(sample))
    print("Выбор случайного города из списка - ", mychoise[len(mychoise)-1])
    mychoise2.append(random.choice(sample))
    print("Выбор случайного города из списка - ", mychoise2[len(mychoise2) - 1])
mychoise.sort()
mychoise2.sort()

plt.figure("AllGraph1", figsize=(13, 4))
plt.subplot(1,3,1)
plt.plot(line, mychoise)

plt.subplot(1,3,2)
plt.plot(line, mychoise2, color='r')

line3 = []
for i in arange(-2.5, 2.5, 0.1):
    line3.append(i)
plt.subplot(1,3,3)
plt.scatter(mychoise, mychoise2, marker='.')
plt.plot(line3, line3,"r--")
plt.show()