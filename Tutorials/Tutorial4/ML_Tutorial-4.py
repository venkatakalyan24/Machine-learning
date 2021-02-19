#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
from matplotlib import pyplot as plt
from numpy.random import normal
from scipy.stats import norm, multivariate_normal
from math import sqrt
np.random.seed(40)



# In[23]:


x = np.linspace(-1, 1, 10)
line = []
for i in x:
    i = -0.3 + 0.5 * i + normal(0, 0.2)
    line.append(i)
plt.scatter(x, line)
plt.plot(x, -0.3 + 0.5 * x, c="green")


# In[24]:


def linear(x, w0, w1):
    return w0 + w1*x
    
def gaussian(y, x, w0, w1):
    sd = 0.2
    constant = sqrt(np.pi * 2) * sd
    return 1/constant * np.exp(-0.5 * ((y - linear(x, w0, w1))/sd) ** 2)

def posterior_sample(s0, m0, phi, beta, t):
    s0Inv = np.linalg.inv(s0)
    Sn = np.linalg.inv(s0Inv + beta * phi.T@phi)
    inner_term = s0Inv@(m0.reshape((2, 1))) + beta * phi.T * t
    Mn = Sn@inner_term
    return Mn.reshape((1, 2))[0], Sn


# In[30]:


w0, w1 = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
prior = multivariate_normal([0, 0], [[0.5, 0], [0, 0.5]])
plt.contourf(w0, w1, prior.pdf(np.dstack((w0, w1))), cmap="jet")


# In[31]:


fig, axs = plt.subplots(10, 3, figsize=(14, 22))
count = 0
prior = prior.pdf(np.dstack((w0, w1)))
m0, s0 = np.array([0, 0]), np.array([[0.5, 0], [0, 0.5]])
beta = 25
x_add, y_add = [], []

for i, j in zip(x, line):
    mle = gaussian(j, i, w0, w1)
    posterior = prior * mle
    prior = posterior
    phi = np.array([1, i]).reshape((1, 2))
    m0, s0 = posterior_sample(s0, m0, phi, beta, j)
    dist = multivariate_normal.rvs(m0, s0, 10)
    x_add.append(i)
    y_add.append(j)
    
    axs[count, 0].contourf(w0, w1, mle, cmap="jet")
    axs[count, 1].contourf(w0, w1, posterior, cmap="jet")
    for lines in dist:
        axs[count, 2].plot(x, lines[0] + lines[1] * x, c="red")
        axs[count, 2].plot(x, m0[0] + m0[1] * x, c="black")
    axs[count, 2].scatter(x_add, y_add)
    count += 1
    fig.savefig("figure-1")


# In[32]:


fig, axes = plt.subplots(1, 3, figsize=(15, 3))
axes[0].contourf(w0, w1, mle, cmap="jet")
axes[0].scatter(-0.3, 0.5, c="black", marker="+")
axes[1].contourf(w0, w1, posterior, cmap="jet")
axes[1].scatter(-0.3, 0.5, c="black", marker="+")
for lines in dist:
    axes[2].plot(x, lines[0] + lines[1] * x, c="red")
    axes[2].plot(x, m0[0] + m0[1] * x, c="black")
axes[2].scatter(x_add, y_add)
fig.savefig("figure-2")


# In[ ]:





# In[ ]:




