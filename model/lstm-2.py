

import numpy as np
import tensorflow as tf
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
import word_preprocessing as wp


processor = wp.Word_Processor('../grammar/')
ml_classes = wp.ml_classes
feature_set, class_set, lexicon_size = processor.words_to_vectors()


lstm_units = 256
batch_size = 5
lstm_layers = 1
learning_rate = 0.001


def split_sets(data, classes, split_point=1):
    split_length = int(data.shape[0] * split_point)
    training_features = data[:split_length]
    training_ops = classes[:split_length]
    testing_features = data[split_length:]
    testing_ops = classes[split_length:]
    return training_features, training_ops, testing_features, testing_ops


# currently using all data for training
trainX, trainY, _, _ = split_sets(feature_set, class_set)
trainX = np.reshape(trainX, newshape=[trainX.shape[0], trainX.shape[1], 1])
print(trainX.shape, trainY.shape)


lstm_net = tflearn.input_data(shape=[None, lexicon_size, 1])
trainY = to_categorical(trainY, nb_classes=len(ml_classes))
#lstm_net = tflearn.embedding(lstm_net)
lstm_net = tflearn.lstm(lstm_net,weights_init='xavier', n_units=lstm_units, return_seq=True)
lstm_net = tflearn.lstm(lstm_net,weights_init='xavier', n_units = lstm_units)
lstm_net = tflearn.fully_connected(
    lstm_net, n_units=len(ml_classes), activation='softmax')
lstm_net = tflearn.regression(lstm_net, optimizer='adam', learning_rate=learning_rate,
                              loss='categorical_crossentropy')

model = tflearn.DNN(lstm_net, tensorboard_verbose=0)
model.fit(trainX, trainY, show_metric=True, batch_size=batch_size, n_epoch=10)
model.save("saved/lstm-model.tfl")
