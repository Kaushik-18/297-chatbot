import numpy as np
import nltk
from nltk.corpus import stopwords
from collections import defaultdict
import random
import pickle

nltk.download('punkt')

from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

ml_classes = ['location', 'professor', 'time', 'professor_office_hours' ,
              'professor_office_location' ,'project_details' ,'syllabus']


class Word_Processor:
    def __init__(self, folder_path):
        self.store = defaultdict(list)
        for ml_class in ml_classes:
            with open(folder_path + "{}.txt".format(ml_class), encoding="utf-8") as file:
                lines = file.read().splitlines()
                for line in lines:
                    self.store[ml_class].append(line)

    # tokenize,lemmatize,stem each sentence
    def preprocess(self, sentence):
        tokens = nltk.word_tokenize(sentence)
        tokens = [t.lower() for t in tokens]
        tokens_filtered = tokens
        stemmed = [stemmer.stem(t) for t in tokens_filtered]
        lemmatized = [lemmatizer.lemmatize(t) for t in stemmed]
        return lemmatized

    ''' convert tokens to vector representation.'''

    def vectorize(self, token_set, all_uniq_words):
        return list(map(lambda x: 1 if x in token_set else 0, all_uniq_words))

    def words_to_vectors(self):
        # represents set of all unique words in our data set
        self.unique_words = set()
        # collect tokenized sentences
        process_sentences = []
        # collect final vector forms
        data_set = []

        for ml_class, sentences in self.store.items():
            for sentence in sentences:
                tokens = self.preprocess(sentence)
                self.unique_words.update(tokens)
                process_sentences.append([ml_class, tokens])

        for ml_class, tokens in process_sentences:
            token_set = set(tokens)
            x = self.vectorize(token_set, self.unique_words)
            y = ml_classes.index(ml_class)
            data_set.append((x, y))

        random.shuffle(data_set)
        features = [data[0] for data in data_set]
        outputs = [op[1] for op in data_set]
        return np.asarray(features), np.asarray(outputs), len(self.unique_words)

    def vectorize_input(self, sentence):
        tokens = self.preprocess(sentence)
        vector = self.vectorize(tokens, self.unique_words)
        return np.asarray(vector)


if __name__ == '__main__':
    procs = Word_Processor('../grammar/')
    train, ops, size = procs.words_to_vectors()
    print(train[:10], ops[:10])
