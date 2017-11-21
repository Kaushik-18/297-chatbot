
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import pickle
import nltk

from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()



# In[2]:


def preprocess(sentence):
    tokens = nltk.word_tokenize(sentence) # tokenize
    tokens = [t.lower() for t in tokens]
    # tokens_filtered = filter(lambda token: token not in stop_words, tokens)
    tokens_filtered = tokens
    stemmed = [stemmer.stem(t) for t in tokens_filtered]
    lemmatized = [lemmatizer.lemmatize(t) for t in stemmed]
    return lemmatized

def vectorize(token_set, all_uniq_words):
    return list(map(lambda x: 1 if x in token_set else 0, all_uniq_words))


# In[3]:


data = pickle.load( open( "training_data", "rb" ) )
all_uniq_words = data['all_uniq_words']
uniq_ml_classes = data['uniq_ml_classes']


# In[4]:


feature_columns=[tf.contrib.layers.real_valued_column(column_name='', dimension=31, default_value=None, dtype=tf.int64, normalizer=None)]
dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units=[10, 10], n_classes=3, feature_columns=feature_columns, model_dir="./chat_model")


# In[7]:


a = "who is the professor"
# a = "where is class?"
# a = "yo when class start eh?"
tokens = preprocess(a)
token_set = set(tokens)
x = vectorize(token_set, all_uniq_words)
test_X = np.array(x).reshape(1, -1)
y_pred = list(dnn_clf.predict(test_X))
print(test_X)
print(y_pred)
print(uniq_ml_classes[y_pred[0]])


# In[ ]:




