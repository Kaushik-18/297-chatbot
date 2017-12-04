import tflearn
import numpy as np
import tensorflow as tf
from tflearn.data_utils import to_categorical, pad_sequences
import word_preprocessing as wp

# creating the lexicon, this is just to load the list of unique words
# TODO we could come up with a better way
processor = wp.Word_Processor('../grammar/')
ml_classes = wp.ml_classes
_, _, lexicon_size = processor.words_to_vectors()


# TODO need a common file to read our hyerparameters
lstm_units = 256
lstm_net = tflearn.input_data(shape=[None, lexicon_size, 1])
#lstm_net = tflearn.embedding(lstm_net)
lstm_net = tflearn.lstm(lstm_net, n_units=lstm_units, return_seq=False)
#lstm_net = tflearn.lstm(lstm_net, n_units = lstm_units)
lstm_net = tflearn.fully_connected(
    lstm_net, n_units=len(ml_classes), activation='softmax')

model = tflearn.DNN(lstm_net, tensorboard_verbose=0)
model.load("saved/lstm-model.tfl")

while True:
    tr_data = input('Enter testing string : ')
    if tr_data != 'q':
        tr_data = processor.vectorize_input(tr_data)
        tr_data = np.reshape(tr_data, newshape=[1, lexicon_size, 1])
        predicted_class = model.predict_label(tr_data)
        predicted_values = model.predict(tr_data)
        print(predicted_class)
        print(predicted_values)
        print(ml_classes[predicted_class[0][0]], predicted_values[0][0])
    else:
        break
