#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import math
from numpy import random

mean,variance = 5,1
sigma = math.sqrt(variance)


# In[2]:


def gaussian(mean,variance,z):
    return 1/(math.sqrt(2*math.pi)*np.sqrt(variance)) * math.exp(-(z - mean)**2/(2*np.sqrt(variance)**2))


# In[3]:


def likelihood(z, mean):
    l = 1
    for i in z:
        l *= gaussian(mean,1,i)
    return l


# In[14]:


z = random.normal(mean,variance, 10)
y = [gaussian(mean,variance,i) for i in z]
plt.plot(z, y, 'ro')
plt.title("Gaussian Distribution")


# In[15]:


likelihoods = [0]*11
for i in range(0, 11):
    likelihoods[i] = likelihood(z, i)
    means[i] = i
plt.plot(means, likelihoods, 'y')
plt.xlabel("Mean")
plt.ylabel("Likelihood")
plt.title("Likelihood curve")


# In[16]:


log_likelihood = [math.log(i) for i in likelihoods]
plt.plot(means,log_likelihood,'b')
plt.xlabel("Mean")
plt.ylabel("Log_Likelihood")
plt.title("Log Likelihood curve")


# In[ ]:





# In[ ]:




