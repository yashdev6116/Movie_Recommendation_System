#!/usr/bin/env python
# coding: utf-8

# In[76]:


import numpy as np
import pandas as pd 


# In[77]:


movies= pd.read_csv('tmdb_5000_movies.csv')
credits= pd.read_csv('tmdb_5000_credits.csv')


# In[78]:


movies.head(1)


# In[79]:


credits.head(1)


# In[80]:


movies=movies.merge(credits,on='title')


# In[81]:


movies.shape


# In[82]:


movies.head(1)


# In[83]:


movies.info()


# In[84]:


#columns to keep
'''
genres
id
keywords
title
overview
cast
crew
'''

movies= movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[85]:


movies.head()


# In[86]:


movies.isnull().sum()


# In[87]:


movies.dropna(inplace=True)


# In[88]:


movies.duplicated().sum()


# In[89]:


movies.iloc[0].genres


# In[90]:


# to bring in form - ['Action','Adventure','Fantasy','SciFi']


# In[91]:


import ast
def process(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
        return L


# In[92]:


movies['genres']=movies['genres'].apply(process)


# In[93]:


movies.head()


# In[94]:


movies['keywords']=movies['keywords'].apply(process)


# In[95]:


movies.head()


# In[96]:


def processcast(obj):
    L=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter !=3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L


# In[99]:


movies['cast']= movies['cast'].apply(processcast)


# In[100]:


movies.head()


# In[101]:


def processdirect(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            L.append(i['name'])
            break
    return L


# In[102]:


movies['crew']=movies['crew'].apply(processdirect)


# In[103]:


movies.head()


# In[105]:


movies['overview']=movies['overview'].apply(lambda x:x.split())


# In[106]:


movies.head()


# In[ ]:


# transform:
'''
Zoe Saldana to ZoeSaldana

'''


# In[111]:


movies['genres']= movies['genres'].apply(lambda x:[i.replace(" ","") for i in x] if x is not None else x)
movies['keywords']= movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x] if x is not None else x)
movies['cast']= movies['cast'].apply(lambda x:[i.replace(" ","") for i in x] if x is not None else x)
movies['crew']= movies['crew'].apply(lambda x:[i.replace(" ","") for i in x] if x is not None else x)


# In[112]:


movies.head()


# In[114]:


movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']


# In[115]:


movies.head()


# movies.info(tags)

# In[120]:


DF=movies[['movie_id','title','tags']]


# In[121]:


DF


# In[126]:


DF['tags'] = DF['tags'].apply(lambda x: " ".join(x) if isinstance(x, list) else str(x))


# In[127]:


DF


# In[128]:


DF.head()


# In[132]:


DF['tags'][0]


# In[130]:


DF['tags']=DF['tags'].apply(lambda x:x.lower())


# In[133]:


DF.head()


# In[ ]:


# our work is to find the similarity between two tags but this in in words that is string so we have to convert it into numbers using vectorization.
#Bag of words- combine all words now it will find most freq words
#remove stop words alse - which is are and or etc...


# In[153]:


import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


# In[154]:


def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# In[155]:


DF['tags']= DF['tags'].apply(stem)


# In[156]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000, stop_words='english')


# In[157]:


vectors=cv.fit_transform(DF['tags']).toarray()


# In[158]:


vectors


# In[160]:


cv.get_feature_names()


# In[149]:


#now we have find distance between thw movies so we will usw coisne 


# In[161]:


from sklearn.metrics.pairwise import cosine_similarity


# In[164]:


similarity=cosine_similarity(vectors)


# In[171]:


sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]


# In[179]:


def recommend(movie):
    movie_index=DF[DF['title']== movie].index[0]
    distances=similarity[movie_index  ]
    movies_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(DF.iloc[i[0]].title)
        


# In[180]:


recommend('Avatar')


# In[182]:


import pickle


# In[185]:


DF['title'].values


# In[186]:


pickle.dump(DF.to_dict(),open('movie_dict.pkl','wb'))


# In[187]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:




