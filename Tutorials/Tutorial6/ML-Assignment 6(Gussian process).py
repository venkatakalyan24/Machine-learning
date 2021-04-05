#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from matplotlib import pyplot as plt
from seaborn import heatmap


# In[2]:


def dataSet_2():
    X_train = np.array([-4, -3, -2, -1, 1]).reshape(-1, 1)
    Y_train = np.sin(X_train)
    X_test = np.arange(-5, 5, 0.2).reshape(-1, 1)
    return X_train,Y_train,X_test


# In[3]:


def kernel(x1, x2, scale=1.0, sigma_f=1.0):
    sqdist = np.sum(x1 ** 2, 1).reshape(-1, 1) + np.sum(x2 ** 2, 1) - 2 * np.dot(x1, x2.T)
    return sigma_f ** 2 * np.exp(-0.5/scale ** 2 * sqdist)


# In[11]:


X_train, Y_train, X_test = dataSet_2()
plt.scatter(X_train,Y_train, color = 'green')


# In[5]:


def plot_gp(mu, cov, X, X_train=None, Y_train=None, samples=[]):
    X = X.ravel()
    mu = mu.ravel()
    uncertainty = 1.96 * np.sqrt(np.diag(cov))
    plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.1)
    plt.plot(X, mu, label='Mean')
    for i, sample in enumerate(samples):
        plt.plot(X, sample, lw=1, ls='--', label=f'Sample {i+1}')
    if X_train is not None:
        plt.plot(X_train, Y_train, 'rx')
    plt.legend()


# In[6]:


beta = 25
C = kernel(X_train, X_train)
k = kernel(X_train, X_test)
c = kernel(X_test, X_test)
C += np.eye(C.shape[0]) * (1/beta)
L = np.linalg.cholesky(C)
alpha = np.dot(np.linalg.inv(L.T), np.dot(np.linalg.inv(L), Y_train))
f = np.dot(k.T, alpha)
v = np.dot(np.linalg.inv(L), k)
var = c - np.dot(v.T, v)


# In[7]:


plot_gp(f, var, X_test, X_train, Y_train)
plt.suptitle("predictive distribustion for function using Gaussian process")


# In[8]:


fig, axes = plt.subplots(2, 2, figsize=(12, 9))
heatmap(C, ax=axes[0, 0])
heatmap(k, ax=axes[0, 1])
heatmap(k.T, ax=axes[1, 0])
heatmap(c, ax=axes[1, 1])


# In[9]:


C_n_plus1 = np.vstack((np.hstack((C, k)), np.hstack((k.T, c))))
heatmap(C_n_plus1)


# In[ ]:




