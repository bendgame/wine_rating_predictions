#!/usr/bin/env python
# coding: utf-8

# In[4]:


from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sqlite3
import category_encoders as ce


import os
import sys
conn = sqlite3.connect('db/wine_data.sqlite')
c = conn.cursor


# In[24]:


df = pd.read_sql("select country,description,rating,price,province,title,variety,winery,color from wine_data where variety = 'Chardonnay'", conn)
df.head(2)


# In[26]:


import re
def add_year(dataframe, dataframe_column):
    l = []
    i = 0 
    for year in range(len(dataframe_column)):
        temp = re.findall(r'\d+', dataframe_column[i]) 
        res = list(map(int, temp)) 
        try: 
            if len(str(res[0])) == 4:
                l.append(res[0])
            elif len(str(res[0])) != 4:
                l.append(0)
        except:
            l.append(0)
        #print(res[0])
        i+=1
    dataframe['year'] = l
    
    return dataframe

add_year(df, df['title'])

def word_count(dataframe, dataframe_column):
    dataframe['word_count'] = dataframe_column.apply(lambda word: len(str(word).split(" ")))
    return df

word_count(df, df['description'])
df.head()


# In[11]:


df['word_count'] = df['description'].apply(lambda word: len(str(word).split(" ")))
df.head()


# In[40]:


labels = ['country','province','title','winery']
numeric= ['price', 'year', 'word_count']
other = ['description', 'variety', 'color']

# encoder = ce.BackwardDifferenceEncoder(cols=[...])
# encoder = ce.BaseNEncoder(cols=[...])
# encoder = ce.BinaryEncoder(cols=[...])
# encoder = ce.CatBoostEncoder(cols=[...])
# encoder = ce.HashingEncoder(cols=[...])
# encoder = ce.HelmertEncoder(cols=[...])
# encoder = ce.JamesSteinEncoder(cols=[...])
# encoder = ce.LeaveOneOutEncoder(cols=[...]) --maybe
# encoder = ce.MEstimateEncoder(cols=[...]) --maybe
# encoder = ce.OneHotEncoder(cols=[...])
# encoder = ce.OrdinalEncoder(cols=[...]) --maybe
# encoder = ce.SumEncoder(cols=[...])
# encoder = ce.PolynomialEncoder(cols=[...])
# encoder = ce.TargetEncoder(cols=[...]) --maybe
# encoder = ce.WOEEncoder(cols=[...]) --binary y value only

x = df[labels]
y = df['price']
ce_ord = ce.TargetEncoder(cols=labels)
enc_df = ce_ord.fit_transform(x, y)


# In[44]:


enc_df['price'] = df['price']
enc_df['word_count'] = df['word_count']
enc_df['year'] = df['year']
enc_df.head()


# In[45]:


from sklearn.model_selection import train_test_split

X = enc_df
y = df['rating']
X_train,X_test, y_train, y_test = train_test_split(X, y, test_size = .3)


# In[ ]:




