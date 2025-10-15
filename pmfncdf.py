#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('precision', '3')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


x_set = np.array([1, 2, 3, 4, 5, 6])
def f(x):
    if x in x_set:
        return x / 21
    else: 
        return 0
X = [x_set, f]


# In[5]:


#확률 p_K를 구한다
prob = np.array([f(x_k) for x_k in x_set])
# x_k와 p_k의 대응을 사전식으로 표시
dict(zip(x_set, prob))


# In[6]:


fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.bar(x_set, prob)
ax.set_xlabel('value')
ax.set_ylabel('probability')

plt.show()


# In[31]:


a = {'사과':1, '딸기':5, '귤':10}
a


# In[30]:


a ={('초콜릿', 200):20, ('마카롱', 500):15, ('쿠키', 300):30}
a


# In[29]:


a = {'사과':1, '딸기':5, '귤':10}
v1 = a['딸기']
v1


# In[11]:


v2=a['레몬']
v2


# In[32]:


f1 = '딸기' in a
f1


# In[33]:


f2 = '레몬' not in a
f2


# In[34]:


f3 = '레몬' in a
f3


# In[35]:


v1 = a.get('딸기')
v1


# In[36]:


v2 = a.get('레몬')
v2


# In[37]:


a = {'초콜릿':1, '마카롱':2, '쿠키':3}
a['초콜릿'] = 'One'
a['마카롱'] = 'Two'
a['쿠키'] = 'Three'
a


# In[38]:


d = dict(초콜릿 = 20, 마카롱 = 15, 쿠키 = 30)
d


# In[39]:


key = ['초콜릿', '마카롱', '쿠키']
value = [20, 15, 30]
d = dict(zip(key, value))
d


# In[40]:


d = dict([('초콜릿', 20), ('마카롱', 15), ('쿠키', 30)])
d


# In[41]:


np.all(prob >= 0 )


# In[42]:


np.sum(prob)


# In[43]:


def F(x):
    return np.sum([f(x_k) for x_k in x_set if x_k <= x])


# In[44]:


F(3)


# In[46]:


y_set = np.array([2 * x_k + 3 for x_k in x_set])
prob = np.array([f(x_k) for x_k in x_set])
dict(zip(y_set, prob))


# In[47]:


np.sum([x_k * f(x_k) for x_k in x_set])


# In[48]:


np.random.choice(5, 5, replace=False)


# In[49]:


np.random.choice(5, 3, replace=False)


# In[50]:


np.random.choice(5, 10)


# In[51]:


np.random.choice(5, 10, p=[0.1, 0, 0.3, 0.6, 0])


# In[52]:


sample = np.random.choice(x_set, int(1e6), p=prob)
np.mean(sample)


# In[54]:


def E(X, g=lambda x: x):
    x_set, f = X
    return np.sum([g(x_k) * f(x_k) for x_k in x_set])
E(X)


# In[55]:


mean = E(X)
np.sum([(x_k-mean) ** 2 * f(x_k) for x_k in x_set])


# In[56]:


def V(X, g=lambda x: x):
    x_set, f = X
    mean = E(X, g)
    return np.sum([(g(x_k)-mean)**2 * f(x_k) for x_k in x_set])


# In[57]:


V(X)


# In[58]:


V(X, lambda x: 2*x + 3)


# In[59]:


2**2 * V(X)


# In[ ]:




