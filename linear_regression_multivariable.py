#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[2]:


df=pd.read_csv("C:/Users/Darshan/Downloads/linear_rg1.csv")


# In[3]:


df.head()


# In[6]:


import math
median_bedrooms=math.floor(df.bedrooms.median()) #in order to fill a 'NaN' ,we have to take a median of this column


# In[5]:


median_bedrooms


# In[9]:


df.bedrooms=df.bedrooms.fillna(median_bedrooms) #'NaN' filled by the median value


# In[10]:


df #preprocessing part is end here,we have to preprocess our data for correct prediction


# In[11]:


reg=linear_model.LinearRegression()
reg.fit(df[['area','bedrooms','age']],df.price)


# In[12]:


reg.coef_


# In[13]:


reg.intercept_


# In[14]:


reg.predict([[3300,5,12]])

