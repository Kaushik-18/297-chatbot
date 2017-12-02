
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import pickle
import nltk
import word_preprocessing as wp

from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()



# In[2]:


data = pickle.load( open( "training_data", "rb" ) )
all_uniq_words = data['all_uniq_words']
uniq_ml_classes = data['uniq_ml_classes']
print(all_uniq_words)
print(uniq_ml_classes)

processor  =  wp.Word_Processor('../grammar/', uniq_ml_classes)


# In[3]:


feature_columns=[tf.contrib.layers.real_valued_column(column_name='', dimension=35, default_value=None, dtype=tf.int64, normalizer=None)]
dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units=[10, 10], n_classes=3, feature_columns=feature_columns, model_dir="./chat_model")


# In[7]:


a = "who is the professor"
# a = "where is class?"
# a = "yo when class start eh?"

x = processor.vectorize(a, all_uniq_words)

test_X = np.array(x).reshape(1, -1)
y_pred = list(dnn_clf.predict(test_X))
print(test_X)
print(y_pred)
print(uniq_ml_classes[y_pred[0]])


# In[5]:


print(all_uniq_words)


# In[ ]:




