#!/usr/bin/env python
# coding: utf-8

# # Binary Logistic Regression

# BinaryLogistic Regression is a classification model.It requries dependent variable to be binary(two possible values), such as exam(pass/fail), sex (male/female), response(yes/no),score(high/low), etc….
#  which is represented by an indicator variable, where the two values are labeled "0" and "1".
# 
# 

# In this project,I build a classifier to classify whether a diabetic or not by training a binary classification model using 
# Logistic Regression. I have used the pima dataset downloaded from the uci machine learning repository website for this project.
# 

# In[1]:


#importing the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


#importing the pima dataset
data= pd.read_csv('D:\\ML_Datasets\\diabetics\\india_pima_diabetics.csv')


# In[3]:


#view dimensons(no of rows and columns) of the dataset
data.shape


# The dataset contains 768 rows and 9 columns

# In[4]:


# preview the dataset
data.head()


# In[5]:


#Description of the data
data.describe()


# In[6]:


#view summary of the dataset
data.info()


# In[7]:


# check missing values in the dataset 
data.isnull().sum()


# In[9]:


#Checking the missing values in the data
sns.heatmap(data.isnull(),cbar=False,cmap="viridis")
plt.show()


# there is no missing values in the dataset

# In[10]:


sns.pairplot(data)


# In[11]:


corr=data.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corr, annot=True, cmap="Greens")
plt.title('Heatmap', fontsize=20)


# In[16]:


# checking the columns of the data
data.columns


# In[13]:


#split the dataset in features and target variable
X=data.iloc[:,:-1]
y=data.iloc[:,-1]
print(X.head(2))
print(y.head(2))


# Spliting the dataset into training and testing for training 80 percent testing 20 percent

# In[14]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)


# In[15]:


# import the class 
from sklearn.linear_model import LogisticRegression
#instantiate the model(using default parameters)
logreg=LogisticRegression()


# In[17]:


#fit the model with data
logreg.fit(X_train,y_train)


# In[18]:


#Predicting the results
y_pred=logreg.predict(X_test)
print(y_pred)


# predict_proba method gives the probabilities for the target variable(0 and 1) in this case, in array form.
# 
# 0 is for probability of no rain and 1 is for probability of rain.
# 
# 

# In[19]:


# probability of getting output as 0 - no rain

logreg.predict_proba(X_test)[:,0].sum()


# In[20]:


# probability of getting output as 1 - rain

logreg.predict_proba(X_test)[:,1].sum()


# In[21]:


#Check accuracy score
from sklearn import metrics
print("Accuracy_score:",metrics.accuracy_score(y_test,y_pred))


# In[22]:


#import the metrics class
from sklearn import metrics
cnf_matrix=metrics.confusion_matrix(y_test,y_pred)
print(cnf_matrix) #26 and 11 are incorrect predictions


# Confusion matrix

# In[23]:


#create heatmap 
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu",fmt='g')


# In[24]:


#Accuracy
print("Accuracy:",metrics.accuracy_score(y_test,y_pred))
print("Precision:",metrics.precision_score(y_test,y_pred))
print("Recall:",metrics.recall_score(y_test,y_pred))


# Roc Curve

# In[25]:


y_pred_proba=logreg.predict_proba(X_test)[::,1]
fpr,tpr,_=metrics.roc_curve(y_test, y_pred_proba)
auc=metrics.roc_auc_score(y_test,y_pred_proba)
plt.plot(fpr,tpr,label="data 1,auc="+str(auc))
plt.legend(loc=1)
plt.show


# the auc(area under curve value is 0.87)






