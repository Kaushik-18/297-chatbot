
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)
import nltk
from nltk.corpus import stopwords
from collections import defaultdict
import random
import pickle
import word_preprocessing as wp

from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


# In[2]:


ml_classes = ['location', 'professor', 'time', 'professor_office_hours',
              'professor_office_location', 'project_details', 'syllabus',
              'exam_details', 'about_the_class', 'greeting', 'feeling', 'farewell',
              'ta', 'fallback']

processor  =  wp.Word_Processor('../grammar/', ml_classes)
train_X, train_Y, all_uniq_words = processor.words_to_vectors()
train_Y = train_Y.reshape(-1,1)

print(train_X.shape)
print(train_Y.shape)


test_X = train_X[-10000:,]
test_Y = train_Y[-10000:,]
train_X = train_X[:-10000, ]
train_Y = train_Y[:-10000, ]

print(train_X.shape)
print(train_Y.shape)
# In[3]:


feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(train_X)
print(feature_columns)
dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units=[100, 100, 100, 50, 50], n_classes=15, feature_columns=feature_columns, model_dir="./chat_model")
dnn_clf.fit(x=train_X, y=train_Y, batch_size=64, steps=24000)
results = dnn_clf.evaluate(x=train_X, y=train_Y)
print('resutls')
print(results)
results2 = dnn_clf.evaluate(x=test_X, y=test_Y)
print(results2)


# In[ ]:


pickle.dump( {'all_uniq_words':all_uniq_words, 'uniq_ml_classes':ml_classes}, open( "training_data", "wb" ) )


# In[ ]:


a = "where is class?"
a = "where is professor office?"
a = "yo when class start eh?"

a = "what is the project"
a = "what is the assignement on?"
a = "who is the president?"
a = "who is the professor"
a = "is it raining today?"
a = "who is stupid?"

x = processor.vectorize(a, all_uniq_words)

test_X = np.array(x).reshape(1, -1)

y_pred = list(dnn_clf.predict(test_X))
y_pred_proba = list(dnn_clf.predict_proba(test_X))
# print(test_X)
print(y_pred)
print(y_pred_proba)
print(ml_classes[y_pred[0]])


# In[ ]:


print(all_uniq_words)


# In[ ]:




