#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


credits=pd.read_csv('tmdb_5000_credits.csv')


# In[3]:


credits.head(5)


# In[4]:


movies=pd.read_csv('tmdb_5000_movies.csv')


# In[5]:


movies.head(5)


# In[6]:


credits_rename=credits.rename(index=str, columns={"movie_id": "id"})


# In[9]:


movies_df=movies.merge(credits_rename, on='id')


# In[12]:


movies_df.head(5)


# In[13]:


movies_cleaned_df = movies_df.drop(columns=['homepage', 'title_x', 'title_y', 'status','production_countries'])
movies_cleaned_df.head()


# In[15]:


movies_cleaned_df.info()


# In[16]:


movies_cleaned_df.head(1)['overview']


# In[17]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[18]:


tfv = TfidfVectorizer(min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3),
            stop_words = 'english')

# Filling NaNs with empty string
movies_cleaned_df['overview'] = movies_cleaned_df['overview'].fillna('')


# In[19]:


tfv_matrix = tfv.fit_transform(movies_cleaned_df['overview'])


# In[20]:


tfv_matrix


# In[21]:


from sklearn.metrics.pairwise import sigmoid_kernel

# Compute the sigmoid kernel
sig = sigmoid_kernel(tfv_matrix, tfv_matrix)


# In[22]:


sig[0]


# In[23]:


# Reverse mapping of indices and movie titles
indices = pd.Series(movies_cleaned_df.index, index=movies_cleaned_df['original_title']).drop_duplicates()


# In[32]:


indices


# In[33]:


indices['Avatar']


# In[34]:


sig[0]


# In[35]:


list(enumerate(sig[indices['Avatar']]))


# In[43]:


sorted(list(enumerate(sig[indices['Newlyweds']])), key=lambda x: x[1], reverse=True)


# In[44]:


def give_rec(title, sig=sig):
    # Get the index corresponding to original_title
    idx = indices[title]

    # Get the pairwsie similarity scores 
    sig_scores = list(enumerate(sig[idx]))

    # Sort the movies 
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)

    # Scores of the 10 most similar movies
    sig_scores = sig_scores[1:11]

    # Movie indices
    movie_indices = [i[0] for i in sig_scores]

    # Top 10 most similar movies
    return movies_cleaned_df['original_title'].iloc[movie_indices]


# In[69]:


result=input("Enter the Movie Name: ")
try:
    print(give_rec(result))
except:
    print("Not Found")


# In[ ]:




