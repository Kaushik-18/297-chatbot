from flask import Flask, request, abort

import numpy as np
import tensorflow as tf
import pickle
import nltk
import word_preprocessing as wp

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


@app.route("/", methods=['POST'])
def hello():
    if not request.json:
        abort(400)
    message = request.json['message']
    x = processor.vectorize(message, all_uniq_words)

    test_X = np.array(x).reshape(1, -1)
    y_pred = list(dnn_clf.predict(test_X))

    prediction = uniq_ml_classes[y_pred[0]]
    return prediction
