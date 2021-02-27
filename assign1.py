import collections
import re
import string

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import tensorflow as tf
# from nltk.corpus import stopwords
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn import metrics


# ADD THE LIBRARIES YOU'LL NEED

'''
About the task:

You are provided with a codeflow- which consists of functions to be implemented(MANDATORY).

You need to implement each of the functions mentioned below, you may add your own function parameters if needed(not to main).
Execute your code using the provided auto.py script(NO EDITS PERMITTED) as your code will be evaluated using an auto-grader.
'''
stopwordsModified = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'only', 'own', 'same', 'so', 'than', 'too', 's', 't', 'can', 'will', 'just', 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'ma']


def f1_score(actual, predicted):
    table = list(zip(actual, predicted))
    freq = collections.Counter(actual)   
    d = {}
    for i in range(1,6):
        for j in range(1,6):
            d[(i,j)] = 0
    for value in table:
        d[value] += 1
    f1 = [0,0,0,0,0]
    t = np.zeros((5,5))
    for i in range(0,5):
        for j in range(0,5):
            t[i][j] = d[(i+1,j+1)]
    sums_h = np.sum(t, axis=0)
    sums_v = np.sum(t, axis = 1)
    prec = [0]*5
    acc = [0]*5
    for i in range(0,5):
        prec[i] = t[i][i]/sums_h[i]
        acc[i] = t[i][i]/sums_v[i]
        f1[i] = (2*prec[i]*acc[i])/(prec[i]+acc[i])
    avg_f1 = 0
    avg = 0
    sum_freq = 0
    for i in range(0,5):
        avg_f1 += freq[i+1]*f1[i]
        sum_freq += freq[i+1]
        avg += f1[i]
    print("Precision:", prec)
    print("Recall:", acc)
    print("F1:", f1)
    print("Macro avg:", avg/5)
    print("Weighted avg:", avg_f1/sum_freq)
    print("Freq:", freq)

def encode_data(embeddings_index, reviews):
    data = np.zeros((len(reviews), 3000), dtype='float32')
    for r in range(len(reviews)):
        for i in range(30):
            if reviews[r][i] not in embeddings_index:
                continue
            data[r][100*i:100*(i+1)] = embeddings_index[reviews[r][i]]
    return data


def convert_to_lower(text):
    # return the reviews after convering then to lowercase
    converted = [x.lower() for x in text]
    return converted


def remove_punctuation(text):
    # return the reviews after removing punctuations
    removed = [re.split(r'\W+', review)[:-1] for review in text]
    return removed


def remove_stopwords(text):
    # return the reviews after removing the stopwords
    # cachedStopWords = stopwords.words("english")
    # print(cachedStopWords)
    processed = [[w for w in review if w not in stopwordsModified] for review in text]
    return processed

def glove_embeddings():
    embedding_path = 'glove.6B.100d.txt'
    embeddings_index = dict()
    f = open(embedding_path)
    for line in f:
        values = line.split()
        word = values[0]
        emb = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = emb
    f.close()
    return embeddings_index
    
def perform_tokenization(text):
    # return the reviews after performing tokenization
    text = [data.split(" ") for data in text]
    # tokenizer = Tokenizer(num_words=15000, filters='@')
    # tokenizer.fit_on_texts(text)
    # tokenized = tokenizer.texts_to_sequences(text)
    return text


def perform_padding(data):
    # return the reviews after padding the reviews to maximum length
    max_length = 30
    reviews = []
    for review in data:
        padded_review = review + [0]*(max_length-len(review))
        reviews.append(padded_review)
    return reviews

def preprocess_data(data):
    reviews = data["reviews"]
    reviews = convert_to_lower(reviews)
    reviews = remove_punctuation(reviews)
    reviews = remove_stopwords(reviews)
    # print(reviews)
    reviews = perform_padding(reviews)
    embeddings_index = glove_embeddings()
    reviews = encode_data(embeddings_index,reviews)
    return reviews


def softmax_activation(x):
    # write your own implementation from scratch and return softmax values(using predefined softmax is prohibited)
    a = tf.exp(x - tf.reduce_max(x, axis=1, keepdims=True))
    s = tf.reduce_sum(a, axis=1, keepdims=True)
    output = a / s
    return output


class NeuralNet:

    def __init__(self):

        self.model = tf.keras.models.Sequential()
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def build_nn(self):
        # add the input and output layer here; you can use either tensorflow or pytorch
        self.model.add(Dense(400, input_shape=(3000,), activation=tf.nn.relu))
        self.model.add(Dense(5, input_shape=(400,), activation=softmax_activation))
        self.model.compile(optimizer='adam', loss=self.loss_fn, metrics=['accuracy'])
        # print(self.model.summary())

    def train_nn(self, batch_size, epochs, data):
        self.reviews = np.asarray(preprocess_data(data))
        self.ratings = np.asarray([rating-1 for rating in data["ratings"]])
        self.model.fit(self.reviews, self.ratings, epochs=epochs, batch_size=batch_size)
        _, train_err = self.model.evaluate(self.reviews, self.ratings, verbose=0)
        print("Train accuracy:", train_err)
        self.model.save('nn_model')

    def predict(self, test_data):
        # return a list containing all the ratings predicted by the trained model
        test_reviews = preprocess_data(test_data)
        if "ratings" in test_data:
            test_ratings = np.asarray([rating-1 for rating in test_data["ratings"]])
            _, test_err = self.model.evaluate(test_reviews, test_ratings, verbose=0)
            print("Test accuracy:",test_err)

        probs = self.model.predict(test_reviews)
        predictions = np.argmax(probs, axis=-1) + 1
        

        print(metrics.classification_report(test_ratings+1, predictions, digits=3))

        return predictions, probs

    def load_model(self):
        self.model = tf.keras.models.load_model('nn_model')

    def test_single(self, text):
        print("Test input:", text)
        test_review = preprocess_data({"reviews": [text]})
        probs = self.model.predict(test_review)
        prediction = np.argmax(probs, axis=-1) + 1
        print("Prediction:", prediction[0])
        print("Probabilities:", probs[0])
        return prediction, probs 

NN = NeuralNet()
NN.build_nn()

# DO NOT MODIFY MAIN FUNCTION'S PARAMETERS
def main(train_file, test_file):

    batch_size, epochs = 200, 75
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    
    NN.train_nn(batch_size,epochs,train_data)
    NN.load_model()
    preds, _ = NN.predict(test_data)
    # if "ratings" in test_data:
    #     f1_score(test_data["ratings"], preds)
    return preds
