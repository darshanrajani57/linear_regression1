#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[10]:


import numpy as np

def gradient_descent(x, y):
    m_curr = 0
    b_curr = 0
    iterations = 1000
    n = len(x)
    learning_rate = 0.001

    for i in range(iterations):
        y_predicted = m_curr * x + b_curr
        cost = (1/n) * sum(val**2 for val in (y - y_predicted))
        
        md = -(2/n) * sum(x * (y - y_predicted))
        bd = -(2/n) * sum(y - y_predicted)
        
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        
        print(f"Iteration {i + 1}: m = {m_curr}, b = {b_curr}, cost = {cost}")

    print(f"Final values: m = {m_curr}, b = {b_curr}, cost = {cost}")

x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 7, 9, 11, 14])

gradient_descent(x, y)

