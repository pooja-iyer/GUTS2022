#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import numpy as np
import pandas as pd
import pickle
import string
# nltk
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
#Keras
import keras
from keras.preprocessing import sequence


# In[2]:


load_model=keras.models.load_model("hate&abusive_model.h5")
with open('tokenizer.pickle', 'rb') as handle:
    load_tokenizer = pickle.load(handle)


# In[3]:


test=pd.read_csv('tweets.csv', header=None, names=['tweet'])
#test.rename(columns ={'tweet'}, inplace = True)
test.head(5)


# In[4]:


stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
             'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
             'being', 'below', 'between','both', 'but', 'by', 'can', 'd', 'did', 'do',
             'does', 'doing', 'down', 'during', 'each','few', 'for', 'from',
             'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
             'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
             'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
             'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
             'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're','s', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such',
             't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
             'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
             'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',
             'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
             'why', 'will', 'with', 'won', 'y', 'you','youu','yous', "youd","youll", "youre",
             "youve", 'your', 'yours', 'yourself', 'yourselves']

STOPWORDS = set(stopwordlist)


# In[5]:


test=pd.read_csv('tweets.csv', header=None, names=['tweet'])
#test.rename(columns ={'tweet'}, inplace = True)
test.head(5)

test0 = test1 = test

english_punctuations = string.punctuation
st = nltk.PorterStemmer()
lm = nltk.WordNetLemmatizer()
tokenizer = RegexpTokenizer(r'w+')

def cleaning_data(data):
    
    data=data.lower().strip()
    #Removing everything other than a-z
    data = re.sub('[^a-z\\s]+', '', data)
         
    #Removing punctuations
    translator = str.maketrans('', '', english_punctuations)
    data = data.translate(translator)
    
    #Removing repeating characters
    data = re.sub(r'(.)1+', r'1', data)
    
    #Removing URLs
    data = re.sub('((www.[^s]+)|(https?://[^s]+))',' ',data)
    
    #Removing stopwords and tokenizing
    d=[]
    words=data.split(" ")
    for w in words:
        if w not in STOPWORDS:
            d.append(w)
    data=d
    
    #Stemming and Lemmatize
    data = [st.stem(word) for word in data]
    data = [lm.lemmatize(word) for word in data]
    return data

test0['tweet'] = test['tweet'].apply(lambda text: cleaning_data(text))
test0.head(10)


# In[9]:


seq = load_tokenizer.texts_to_sequences(test0['tweet'])
padded = sequence.pad_sequences(seq, maxlen=300)
print("seq",seq)
pred =[]
hate=0
no_hate=0
pred = load_model.predict(padded)
print("pred", pred)
for p in pred:
    if p<0.5:
        no_hate+=1
        print("no hate")
    else:
        hate+=1
        print("hate and abusive")


# In[12]:


print("Total number of hate tweets:",hate)
print("Total number of non-hate tweets:",no_hate)
print("Hate & Abusive :",(hate/(hate+no_hate))*100,"%")
print("No Hate :",(no_hate/(hate+no_hate))*100,"%")

