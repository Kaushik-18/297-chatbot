from flask import Flask, request, abort

import numpy as np
import tensorflow as tf
import pickle
import nltk
import word_preprocessing as wp
import random
from urllib.parse import urlencode
import urllib.request
import string
translator=str.maketrans('','',string.punctuation)

from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

app = Flask(__name__)

data = pickle.load( open( "training_data", "rb" ) )
all_uniq_words = data['all_uniq_words']
uniq_ml_classes = data['uniq_ml_classes']

processor  =  wp.Word_Processor('../grammar/', uniq_ml_classes)


feature_columns=[tf.contrib.layers.real_valued_column(column_name='', dimension=912, default_value=None, dtype=tf.int64, normalizer=None)]
dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units=[100, 100, 100, 50, 50], n_classes=15, feature_columns=feature_columns, model_dir="./chat_model")

answers = { 'location': ['The class meets at Health Building 407', 'Health Building 407'], 'professor': ['Professor Simon Shim', 'The class is being taught by Professor Simon Shim'], 'time': ['Every Tuesday 3 to 6 PM'], 'professor_office_hours': ['Monday 2:30 to 4:00 PM'], 'professor_office_location': ['Engineering building room 269'], 'project_details': ['The final project is to create an highly intelligent chatbot'], 'syllabus': ['The class covers all the deep learning architechtures such as DNN, CNN, RNN in detail'], 'exam_details': ['There is a mid term and final exam.', 'The final exam is on December 14th'], 'about_the_class': ['The class is all learning the latest deep learning techniques'], 'greeting': ['Hey!', 'Hello!', 'Hi!'], 'feeling': ['I am good. How are you?'], 'ta': ['Abhiram and Srivatsa are the teaching assistants for the course']}


@app.route("/", methods=['POST'])
def hello():
    try:
        if not request.json:
            abort(400)
        message = request.json['message']
    
    
        x = processor.vectorize(message, all_uniq_words)
    
        test_X = np.array(x).reshape(1, -1)
        y_pred = list(dnn_clf.predict(test_X))
    
        prediction = uniq_ml_classes[y_pred[0]]
        if prediction in answers.keys():
            return random.choice(answers[prediction])
        else:
            message = message.translate(translator)
            qs = urlencode({'ss': message.lower()})
            return urllib.request.urlopen("http://34d86b90.ngrok.io/ss?" + qs).read()
    except:
        return "Sorry, I did not understand!"
