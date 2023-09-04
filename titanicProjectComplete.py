#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[19]:


# Loading the data 
titanic_data = pd.read_csv('C:/Users/cuteb/OneDrive/Documents/titanic/tested.csv')


# In[20]:


titanic_data.head()


# In[21]:


titanic_data.shape


# In[22]:


# Information about the data
titanic_data.info()


# In[23]:


# Finding number of missing values
titanic_data.isnull().sum()


# # Taking care of missing values

# In[24]:


# Dropping cabin column
titanic_data = titanic_data.drop(columns='Cabin', axis=1)


# In[25]:


# Replacing missing values of Age with mean values
titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)


# In[26]:


# Replacing missing values of Fare with mean values
titanic_data['Fare'].fillna(titanic_data['Fare'].mean(), inplace=True)


# In[27]:


titanic_data.isnull().sum()


# # Data Analysis

# In[28]:


# Statistics
titanic_data.describe()


# In[ ]:





# # Data Visualization

# In[29]:


sns.set()


# In[30]:


# Finding surival count, 0 represents not survived and 1 represends survived
titanic_data['Survived'].value_counts()


# In[31]:


# Count plot for "Survived" column
sns.countplot(x='Survived', data=titanic_data)


# In[32]:


# Finding sex count
titanic_data['Sex'].value_counts()


# In[33]:


# Count plot for "Sex" column
sns.countplot(x='Sex', data=titanic_data)


# In[34]:


# Survivers based on their gender
sns.countplot(x="Sex", hue="Survived", data=titanic_data)


# In[35]:


# Count plot for "Pclass" column
sns.countplot(x='Pclass', data=titanic_data)


# In[36]:


# Survivers based on their ticket class
sns.countplot(x="Pclass", hue="Survived", data=titanic_data)


# # Changing into numerical colums

# In[37]:


titanic_data['Sex'].value_counts()


# In[38]:


titanic_data['Embarked'].value_counts()


# In[39]:


# Converting caterogies into numberical
titanic_data.replace({'Sex': {'male': 0, 'female': 1}, 'Embarked': {'S': 0, 'C': 1, 'Q': 2}}, inplace=True)


# In[40]:


titanic_data.head()


# # Separation

# In[41]:


X = titanic_data.drop(columns = ['PassengerId', 'Name', 'Ticket', 'Survived'], axis=1)
Y = titanic_data['Survived'] #Storing survived column's data in a new variable 


# In[42]:


print(X)


# In[43]:


print(Y)


# # Training and Testing data

# In[44]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state=2)


# In[45]:


print(X.shape, X_train.shape, X_test.shape)


# # Model training

# In[46]:


mod = LogisticRegression()


# In[47]:


mod.fit(X_train, Y_train)


# In[48]:


# Accuracy on training data
X_train_pred = mod.predict(X_train)
print(X_train_pred)


# In[53]:


training_data_accuracy = accuracy_score(Y_train, X_train_pred)
print("Training data accuracy: ", training_data_accuracy)


# In[54]:


# Accuracy on test data
X_test_pred = mod.predict(X_test)
print(X_test_pred)


# In[55]:


test_data_accuracy = accuracy_score(Y_test, X_test_pred)
print("Test data accuracy: ", test_data_accuracy)


# In[ ]:




