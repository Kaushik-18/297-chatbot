
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
import word_preprocessing
from keras.utils.np_utils import to_categorical
import random
from keras.preprocessing.text import Tokenizer


word_set = word_preprocessing.Word_Processor('../grammar/').store
class_size = len(word_set.keys())
key_list = list(word_set.keys())

sentence_set = []
all_sentences = []
for class_key, values in word_set.items():
    key_index = key_list.index(class_key)
    for sentence in values:
        sentence_set.append([key_index, sentence])
        all_sentences.append(sentence)

random.shuffle(sentence_set)
# create one hot encoding
lexicon_size = 50
#tokenizer = Tokenizer(nb_words=lexicon_size, split=' ')
#tokenizer.fit_on_texts(all_sentences)
test_encoding = [one_hot(sentence[1], lexicon_size)
                 for sentence in sentence_set]
# test_encoding = [tokenizer.texts_to_sequences(sentence[1])
#                for sentence in sentence_set]

test_labels = [label[0] for label in sentence_set]
test_encode_labels = to_categorical(test_labels, num_classes=class_size)

# max words in a sentence
max_sentence_length = 10
padded_test_encoding = pad_sequences(test_encoding, maxlen=10)

# Define the network
embedding_length = 128
model = Sequential()
model.add(Embedding(input_dim=lexicon_size,
                    output_dim=embedding_length, 
                    input_length=max_sentence_length))
model.add(LSTM(196))
model.add(Dense(units=class_size))
model.compile(optimizer='adam',
              loss='categorical_crossentropy', metrics=['acc'])
print(model.summary())
''' model.fit(padded_test_encoding, test_encode_labels, epochs=200, verbose=1)
loss, accuracy = model.evaluate(
    padded_test_encoding, test_encode_labels, verbose=1)
print('Accuracy: %f' % (accuracy * 100)) '''
#classes = model.predict(x_test, batch_size=128)


print(test_encode_labels)
while True:
    ip = input("Enter Test string ")
    #ip_encode = tokenizer.texts_to_sequences(ip)
    ip_encode = one_hot(ip, n=lexicon_size)
    classes = model.predict_classes(x=ip_encode)
    print(classes)
    prediction = model.predict(ip_encode)
    print(prediction)
