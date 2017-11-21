
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import nltk
from nltk.corpus import stopwords
from collections import defaultdict
import random
import pickle

from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


# In[2]:


ml_classes = ['location', 'professor', 'time']
store = defaultdict(list)
for ml_class in ml_classes:
    with open("{}.txt".format(ml_class), encoding="utf-8") as file:
        lines = file.read().splitlines()
        for line in lines:
            store[ml_class].append(line)

stop_words = set(stopwords.words('english'))


all_words = []
sentences = []

def preprocess(sentence):
    tokens = nltk.word_tokenize(sentence) # tokenize
    tokens = [t.lower() for t in tokens]
    # tokens_filtered = filter(lambda token: token not in stop_words, tokens)
    tokens_filtered = tokens
    stemmed = [stemmer.stem(t) for t in tokens_filtered]
    lemmatized = [lemmatizer.lemmatize(t) for t in stemmed]
    return lemmatized


for ml_class, lines in store.items():
    for line in lines:
        processed_tokens = preprocess(line)
        all_words.extend(processed_tokens)
        sentences.append((processed_tokens, ml_class))


# In[3]:


all_uniq_words = list(set(all_words))
uniq_ml_classes = list(set(ml_classes))

def vectorize(token_set, all_uniq_words):
    return list(map(lambda x: 1 if x in token_set else 0, all_uniq_words))

training = []

for tokens, ml_class in sentences:
    token_set = set(tokens)
    x = vectorize(token_set, all_uniq_words)
    y = uniq_ml_classes.index(ml_class)
    training.append([np.array(x), np.array([y])])

random.shuffle(training)
train_set = np.asarray(training)

train_X = train_set[:, 0]
train_X = np.vstack(train_X)
print(train_X.shape)

train_Y = train_set[:, 1]
train_Y = np.vstack(train_Y)
print(train_Y.shape)

print(train_X[:1])


# In[5]:


feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(train_X)
print(feature_columns)
dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units=[10, 10], n_classes=3, feature_columns=feature_columns, model_dir="./chat_model")
dnn_clf.fit(x=train_X, y=train_Y, steps=20000)


# In[6]:


pickle.dump( {'all_uniq_words':all_uniq_words, 'uniq_ml_classes':uniq_ml_classes}, open( "training_data", "wb" ) )


# In[ ]:




