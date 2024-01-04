#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df=pd.read_csv("C:/Users/Darshan/Downloads/homeprices.csv")


# In[3]:


df


# In[4]:


dumies=pd.get_dummies(df.town) #here the value of town is given in the form of data means numerical value is not given so we have to convert in numerical value and then we use that data.machine learning is useful for numerical value
dumies


# In[5]:


merged=pd.concat([df,dumies],axis='columns')
merged


# In[6]:


final=merged.drop(['town','west windsor'],axis='columns')
final #we need to remove 'town' column and one of the dummy variable column because the rule of dummy variable trap problem(because multicolinearity variable(all variables are derived from one variable)) -> this is not mendatory for linear regression model beacuse this model automatically remove column from dummy variable


# In[7]:


from sklearn.linear_model import LinearRegression
model=LinearRegression()


# In[8]:


x=final.drop(['price'],axis='columns') #price is a dependent variable


# In[9]:


x


# In[10]:


y=final.price


# In[11]:


y


# In[12]:


model.fit(x,y)


# In[13]:


model.predict([[2800,0,1]])


# In[ ]:




