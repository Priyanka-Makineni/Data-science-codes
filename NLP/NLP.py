#!/usr/bin/env python
# coding: utf-8

# ## Natural Language Processing

# #### Definition

# Natural language processing (NLP) is a subfield of linguistics, computer science, information engineering, and artificial intelligence concerned with the interactions between computers and human (natural) languages, in particular how to program computers to process and analyze large amounts of natural language data.
# 
# 

# ### Data set information

# Restaurant reviews dataset:- <br>This dataset contains two features(columns)as Review and Liked ,1000 rows  and 2000 thousand datapoints.

# In[24]:


# importing nltk library
import nltk


# ### Reading and Exploring the dataset

# In[25]:


# Reading the raw data
rawdata=open("Restaurant_Reviews.tsv").read()
rawdata[1:1001]


# In[26]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[27]:


# Importing the dataset
data=pd.read_csv("D:\\mail downloads\\NLP\\Restaurant_Reviews.tsv",delimiter="\t",quoting=3)
data.head(2)


# ### Pre-Processing the Data

# #### 1. Remove the punctuation Marks<br>
# For more Information click here.[punctuation](https://en.wikipedia.org/wiki/Punctuation)<br>
#  It removes The all Punctuation marks(!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~)
# 

# In[28]:


import string
print(string.punctuation)


# The above are the all punctuation marks

# In[29]:


# Function to remove punctuation
def remove_punct(text):
    text_nopunct="".join([char for char in text if char not in string.punctuation]) # It will discard all punctuation
    return text_nopunct
data["Review_clean"]=data["Review"].apply(lambda x:remove_punct(x))
data.head(2)


# #### 2. Tokenization<br>
# It Splits the Sentences into **words**.<br>
# For more informattion click here.[Tokenization](https://www.guru99.com/tokenize-words-sentences-nltk.html)

# In[30]:


import re
# Function to Tokenize  words
def tokenize(text):
    tokens=re.split('\W+', text) # w+ means that either a word character (A-Z,a-z,0-9)
    return tokens
data["Review_tokenized"]=data["Review_clean"].apply(lambda x:tokenize(x))
data.head(2)


# #### 3. Remove Stop words<br>
# It Removes the **Stop Words**.<br>
# For more information click here[Stop Words](https://www.geeksforgeeks.org/removing-stop-words-nltk-python/).

# In[31]:


import nltk
stopwords=nltk.corpus.stopwords.words("english") # All englishwords


# In[32]:


# Function to remove stopwords
def remove_stopwords(tokenized_list):
    text=[word for word in tokenized_list if word not in stopwords] # To remove all stopwords
    return text
data["Review_nostop"]=data["Review_tokenized"].apply (lambda x:remove_stopwords(x))
data.head(2)


# ### Stemming<br>
# It reduce a word to its **stem form**.It removes suffices, like “ing”, “ly”, “s”, etc. by a simple rule-based approach. It reduces the corpus of words but often the actual words get neglected. eg: Entitling,Entitled->Entitl<br>
# For more information [Stemming](https://www.geeksforgeeks.org/python-stemming-words-with-nltk/)

# In[33]:


ps=nltk.PorterStemmer()
def stemming(tokenized_text):
    text=[ps.stem(word) for word in tokenized_text]
    return text
data["Review_stemmed"]=data["Review_nostop"].apply(lambda x:stemming(x))
data.head(2)


# #### Lemmatizing<br>
# It derives the canonical form (‘lemm).a’) of a word. i.e the root form. It is better than stemming as it uses a dictionary-based approach i.e a morphological analysis to the root word.eg: Entitling, Entitled->Entitle.<br>
# For more information [Lemmatization](https://www.machinelearningplus.com/nlp/lemmatization-examples-python/).
# 

# nltk.download()<br>
# In order to download **wordnet** and different packages for different **NLP** using the above code
# 

# In[34]:


wn=nltk.WordNetLemmatizer()
 
def lemmatizing(tokenized_text):
    text=[wn.lemmatize(word) for word in tokenized_text]
    return text
data["Review_lemmatized"]=data["Review_nostop"].apply(lambda x:lemmatizing(x))
data.head(2)


# ### vectorizing the data: Bag of words<br>
# You need to convert these text into some numbers or vectors of numbers. Bag-of-words model(BoW ) is the simplest way of extracting features from the text. BoW converts text into the matrix of occurrence of words within a document.
# 

# In[35]:


from sklearn.feature_extraction.text import CountVectorizer
count_vect=CountVectorizer()
X_counts=count_vect.fit_transform(data["Review"])
print(X_counts.shape)
#print(count_vect.get_features_names())


# ### Vectorizing data:N-grams<br>
# It is a set of co-occurring or continuous sequence of n items from a sequence of large text or sentence. The item here could be words, letters, and syllables. 1-gram is also called as unigrams are the unique words present in the sentence. Bigram(2-gram) is the combination of 2 words. Trigram(3-gram) is 3 words and so on.

# In[36]:


from sklearn.feature_extraction.text import CountVectorizer
ngram_vect=CountVectorizer(ngram_range=(2,2))
X_counts=ngram_vect.fit_transform(data['Review'])
print(X_counts.shape)
print(ngram_vect.get_feature_names())


# #### Vectorizing Data:TF-IDF<br>
# It computes “relative frequency” that a word appears in a document compared to its frequency across all documents. It is more useful than “term frequency” for identifying “important” words in each document (high frequency in that document, low frequency in other documents).<br>
# NLTK does not support tf-idf. So, we're going to use scikit-learn. The scikit-learn has a built in tf-Idf implementation while we still utilize NLTK's tokenizer and stemmer to preprocess the text.

# In[37]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vect=TfidfVectorizer()
X_tfidf=tfidf_vect.fit_transform(data['Review'])
print(X_tfidf.shape)
print(tfidf_vect.get_feature_names())
 


# #### Feature Creation<br>
# Feature engineering is the process of using domain knowledge of the data to create features that make machine learning algorithms work. It is like an art as it requires domain knowledge and it can tough to create features, but it can be fruitful for ML algorithm to predict results as they can be related to the prediction.
# 

# In[38]:


import string
# Function to calculate length of message excluding space
data["Review_len"]=data["Review"].apply(lambda x: len(x)-x.count(" "))
#print(dataset.head(2))

def count_punct(text):
    count=sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text)-text.count(" ")),3)*100
data["punct%"] = data["Review"].apply(lambda x:count_punct(x))
data.head(2)


# In[39]:


bins=np.linspace(0,200,40)
plt.hist(data[data['Liked']==1]['Review_len'],bins,alpha=0.5,normed=True,label=1)
plt.hist(data[data['Liked']==0]['Review_len'],bins,alpha=0.5,normed=True,label=0)
plt.legend(loc="upper right")
plt.show()


# In[40]:


bins=np.linspace(0,50,40)
plt.hist(data[data['Liked']==1]['punct%'],bins,alpha=0.5,normed=True,label=1)
plt.hist(data[data['Liked']==0]['punct%'],bins,alpha=0.5,normed=True,label=0)
plt.legend(loc="upper right")
plt.show()


# #### Model Selection<br>
# We use an ensemble method of machine learning where multiple models are used and their combination produces better results than a single model(Support Vector Machine/Naive Bayes). Ensemble methods are the first choice for many Kaggle Competitions. Random Forest i.e multiple random decision trees are constructed and the aggregates of each tree are used for the final prediction. It can be used for classification as well as regression problems. It follows a bagging strategy where randomly.
# 
# **Grid-search**: It exhaustively searches overall parameter combinations in a given grid to determine the best model.
# 
# 

# In[41]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
rf=RandomForestClassifier()
param={'n_estimators':[10,150,300],'max_depth':[30,60,90,None]}
gs=GridSearchCV(rf,param,cv=5,n_jobs=-1)# For parallelizing the speech
gs_fit=gs.fit(X_counts,data['Liked'])
pd.DataFrame(gs_fit.cv_results_).sort_values('mean_test_score',ascending=False).head()


# In[42]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
rf=RandomForestClassifier()
param={'n_estimators':[10,150,300],'max_depth':[30,60,90,None]}
gs=GridSearchCV(rf,param,cv=5,n_jobs=-1)# For parallelizing the speech
gs_fit=gs.fit(X_tfidf,data['Liked'])
pd.DataFrame(gs_fit.cv_results_).sort_values('mean_test_score',ascending=False).head()


# In[43]:


corpus=X_counts
X = corpus.toarray()
y = data.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
gs.fit(X_train, y_train)

# Predicting the Test set results
y_pred = gs.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# In[44]:


print(cm)


# In[45]:


import seaborn as sns
sns.heatmap(cm,annot=True)


# In[46]:


from sklearn.metrics import classification_report
cr=classification_report(y_test,y_pred)
print(cr)


# ## Conclusion

# Hence we get 65% Accuracy by using Random Forest Classifier due to the uniqueness we cant get more than 60% Accuracy in order to improve we need to use rating system that far improves the information in our training data so that we can get more Accuracy 

# In[ ]:




