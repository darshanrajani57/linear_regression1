#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df=pd.read_csv("C:/Users/Darshan/Downloads/carprices.csv")


# In[3]:


df.head()


# In[4]:


import matplotlib.pyplot as plt


# In[5]:


plt.scatter(df['Mileage'],df['Sell Price($)'])


# In[6]:


plt.scatter(df['Age(yrs)'],df['Sell Price($)'])


# In[10]:


x=df[['Mileage','Age(yrs)']]
y=df['Sell Price($)']


# In[11]:


x


# In[12]:


y


# In[13]:


from sklearn.model_selection import train_test_split


# In[14]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[15]:


len(x_train)


# In[16]:


len(x_test)


# In[17]:


x_train


# In[18]:


from sklearn.linear_model import LinearRegression
clf=LinearRegression()


# In[20]:


clf.fit(x_train,y_train)


# In[21]:


clf.predict(x_test)


# In[22]:


y_test


# In[23]:


clf.score(x_test,y_test)


# In[ ]:




