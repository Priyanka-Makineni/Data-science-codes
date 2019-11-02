#!/usr/bin/env python
# coding: utf-8

# ## Classificaton Models

# ### Definition

# In Machine Learning a classification model tries to draw some conclusion from the input values given for training.
# It will predict the class labels/categories for the new data.
# The data set may simply be bi-class (like identifying whether the person is male or female or that the mail is spam or non-spam) or it may be multi-class too. Some examples of classification problems are: speech recognition, handwriting recognition, bio metric identification, document classification etc.
# 
# 

# ## Data set Information

# I am using the Glass Dataset which consists of 214 rows,11 columns and 2354 observations.
# The study of classification of types of glass was motivated by criminological investigation. At the scene of the crime, the glass left can be used as evidence...if it is correctly identified!
# 
# 

# ## Attribute Information:
# 
# 1. Id number: 1 to 214
# 2. RI: refractive index
# 3. Na: Sodium (unit measurement: weight percent in corresponding oxide, as are attributes 4-10)
# 4. Mg: Magnesium
# 5. Al: Aluminum
# 6. Si: Silicon
# 7. K: Potassium
# 8. Ca: Calcium
# 9. Ba: Barium
# 10. Fe: Iron
# 11. Type of glass: (class attribute)<br>
# -- 1 building_windows_float_processed <br>
# -- 2 building_windows_non_float_processed  <br>
# -- 3 vehicle_windows_float_processed  <br>
# -- 4 vehicle_windows_non_float_processed (none in this database)  <br>
# -- 5 containers  <br>
# -- 6 tableware  <br>
# -- 7 headlamps  <br>
# 
# 

# In[1]:


#Importing the necesserary Libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# importing the dataset
df=pd.read_csv("D:\\ML_Datasets\\Glass\\glass.csv")


# In[3]:


# Preview the data
df.head()


# ## Exploratory Data Analysis

# In[4]:


# Shape of the data i.e, no of rows and columns
df.shape


# Glass dataset contains 214 rows and 11 columns

# In[5]:


#data information  i.e, data types of different columns
df.info()


# In[6]:


# Description of the data
df.describe()


# In[7]:


# checking missing values in the dataset
df.isnull().sum()


# In[8]:


sns.heatmap(df.isnull(),cbar=False,cmap="viridis")


# There is no missing values in the dataset

# In[9]:


sns.pairplot(df)


# In[10]:


corr=df.corr()


# In[11]:


sns.heatmap(corr,annot=True,cmap="Greens")
plt.figure(figsize=(15,15))
plt.show()


# In[12]:


# For checking of outliers in the dataset
sns.boxplot(df,orient='v')


#  In the above diagram we see that there are so many outliers.so we have to normalize the dataset.
#  Normalization means to scale a variable to have a values between 0 and 1,

# #### Assigning the dataset to the x and y variables

# In[13]:


x=df.iloc[:,:-1]
y=df.iloc[:,-1]


# In[14]:


# sample of x and y variables
print(x.sample(2))
print(y.sample(2))


# #### Normalizing the datasset using formula X=(X-min(x)-max(X)-min(X))
# 

# In[15]:


x=((x-np.min(x))/((np.max(x)-np.min(x))))


# In[16]:


# sample of x value after noramlization
print(x.shape)
print(x.sample(5))


# ### Training the Algorithms

# In[17]:


# spliting the data into training and testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)


# In[18]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[19]:


# Importing packages for fitting different models to the data
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier



# Importing functions to get the model fitting for the data 
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix  
from sklearn.model_selection import cross_val_score


# In[20]:


# Fitting the all models
svc = SVC(kernel="rbf", random_state=0)
rfc = RandomForestClassifier(random_state=0)
dct=DecisionTreeClassifier(random_state=0)
nb=GaussianNB()
knn=KNeighborsClassifier(n_neighbors=5)
sgd=SGDClassifier(loss='hinge',shuffle=True,random_state=0)
classifier=LogisticRegression(random_state=0)


classifier.fit(x_train,y_train)
svc.fit(x_train,y_train)
rfc.fit(x_train,y_train)
nb.fit(x_train,y_train)
sgd.fit(x_train,y_train)
dct.fit(x_train,y_train)
knn.fit(x_train,y_train)


# #### Making Predictions

# In[21]:


#Predicting the test set results
y_pred = classifier.predict(x_test)
y_pred_svc = svc.predict(x_test)
y_pred_rfc = rfc.predict(x_test)
y_pred_nb = nb.predict(x_test)
y_pred_dct = dct.predict(x_test)
y_pred_knn = knn.predict(x_test)
y_pred_sgd = sgd.predict(x_test)


# #### Evaluating the Models

# In[22]:


#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
cm_svm = confusion_matrix(y_test,y_pred_svc)
cm_rfc = confusion_matrix(y_test,y_pred_rfc)
cm_nb = confusion_matrix(y_test,y_pred_nb)
cm_dct = confusion_matrix(y_test,y_pred_dct)
cm_knn = confusion_matrix(y_test,y_pred_knn)
cm_sgd = confusion_matrix(y_test,y_pred_sgd)

print('***confusion matrix for multi Logistic Regression*** :')
print(cm)
print('***confusion matrix for SVM ***:')
print(cm_svm)
print('***confusion matrix for Decision Tree *** :')
print(cm_dct)
print('*** confusion matrix for Random Forest *** :')
print(cm_rfc)
print('*** confusion matrix for Naive Bayes *** :')
print(cm_nb)
print('*** confusion matrix for KNN *** :')
print(cm_knn)
print('*** confusion matrix for Stochastic Gradient Descent *** :')
print(cm_sgd)


# In[23]:


# Classification Report for all models
from sklearn.metrics import classification_report
print('classification Report for multi Logistic Regression:')
print(classification_report(y_test,y_pred))
print('classification Report for SVM :')
print(classification_report(y_test,y_pred_svc))
print('classification Report for Random Forest:')
print(classification_report(y_test,y_pred_rfc))
print('classification Report for Naive Bayes:')
print(classification_report(y_test,y_pred_nb))
print('classification Report for Decision Tree:')
print(classification_report(y_test,y_pred_dct))
print('classification Report for KNN:')
print(classification_report(y_test,y_pred_knn))
print('classification Report for Gradient Boosted Machine:')
print(classification_report(y_test,y_pred_sgd))


# In[24]:


# Accuracy for all models
from sklearn import metrics
amlr = metrics.accuracy_score(y_test,y_pred)
asvc = metrics.accuracy_score(y_test,y_pred_svc)
arfc = metrics.accuracy_score(y_test,y_pred_rfc)
adct = metrics.accuracy_score(y_test,y_pred_dct)
aknn = metrics.accuracy_score(y_test,y_pred_knn)
asgd = metrics.accuracy_score(y_test,y_pred_sgd)
anb = amlr = metrics.accuracy_score(y_test,y_pred_nb)

print('Accuracy of the Multi class Logistic Regression:')
print(amlr)
print(' Accuracy of the SVM :')
print(asvc)
print(' Accuracy of the Random Forest :')
print(arfc)
print(' Accuracy of the Decision Tree :')
print(adct)
print('Accuracy of the KNN :')
print(aknn)
print('Accuracy of the Gradient Boosted Machine:')
print(asgd)
print(' Accuracy of the Naive Bayes :')
print(anb)


# In[25]:


accuracy_dict = {'MLR' : amlr, 'SVC' : asvc, 'RFC': arfc, 'DCT': adct, 'NB': anb, 'SGD': asgd, 'KNN' : aknn}
print('Max Accuracy', max(accuracy_dict.items(), key=lambda k: k[1]))
print('Min Accuracy', min(accuracy_dict.items(), key=lambda k: k[1]))


# ### Conclusion

# From the above all models  Decision Tree gives Maximum Accuracy and Gradient boosted  gives minimum  Accuracy

# In[ ]:




