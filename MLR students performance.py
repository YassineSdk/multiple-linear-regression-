#!/usr/bin/env python
# coding: utf-8

# In[41]:


import pandas as pd
import numpy as np 
import seaborn as sns
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy import stats 


#  the Student Performance Dataset is a dataset designed to examine the factors influencing academic student performance. The dataset consists of 10,000 student records, with each record containing information about various predictors and a performance index the target 
#  so in our case study we will apply linear regression models to under cover the factors that determines the academic performance by analysing the relation between this predictors and finding trends and linear relation

# # 1-cleaning and preparing the data set 

# In[3]:


df = pd.read_csv("Student_Performance.csv")


# In[4]:


df.head()


# In[32]:


df = pd.get_dummies(df,columns=["exta_activites"], drop_first=True, dtype=int)


# In[14]:


df.isnull().sum()


# In[34]:


df.head()


# In[10]:


df.dtypes


# In[12]:


df.notnull().sum()


# In[22]:


df.columns


# In[29]:


df.rename(columns={"Hours Studied" :"H_study"},inplace=True)
df.rename(columns={"Extracurricular Activities":"exta_activites"},inplace=True)
df.rename(columns={"Sample Question Papers Practiced":"SQuetion_Papers_practiced"},inplace=True)


# In[31]:


df.columns


# # 2-understanding the relation between the quantitive varibles and the target variable

# In[35]:


df.corr()


# In[40]:


plt.figure(figsize=(10, 10))
sns.regplot(x = "Previous Scores" , y = "Performance Index",data = df)
plt.ylim(0.)


# In[39]:


plt.figure(figsize=(10,10))
sns.regplot(x ="H_study" ,y = "Performance Index" , data = df )
plt.ylim(0.)


# In[44]:


pearson_coef,p_value = stats.pearsonr(df['H_study'],df['Performance Index'])
print("the pearson coeficient is ",pearson_coef ,"and the p-value is " , p_value)


# as we can see they is a moderate linear relationship btw the ['H_study'] ,['Performance Index'] r=0.4 , r~0.5 with a p_value of 0 which means the positif linear relationship is statistcaly significant 

# In[45]:


pearson_coef,p_value = stats.pearsonr(df["Previous Scores"] , df["Performance Index"])
print("the pearson coeficient is ",pearson_coef ,"and the p-value is " , p_value)


# as we can see they is a very strong linear relationship btw the ["Previous Scores"] ,['Performance Index'] r=0.9 , r~1 with a p_value of 0 which means the positif linear relationship is statistcaly significant 
