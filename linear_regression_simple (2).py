#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[8]:


df=pd.read_csv("C:/Users/Darshan/Downloads/linear_rg1.csv")


# In[9]:


df.head()


# In[24]:


plt.xlabel('area(sqft)')
plt.ylabel('price(Rs.)')
plt.scatter(df.area,df.price)


# In[15]:


reg=linear_model.LinearRegression()
reg.fit(df[['area']],df.price)


# In[20]:


reg.predict([[3300]])


# In[21]:


reg.coef_


# In[22]:


reg.intercept_


# In[29]:


plt.xlabel('area(sqft)')
plt.ylabel('price(Rs.)')
plt.scatter(df.area,df.price)
plt.plot(df.area,reg.predict(df[['area']]),color='blue')


# In[ ]:




