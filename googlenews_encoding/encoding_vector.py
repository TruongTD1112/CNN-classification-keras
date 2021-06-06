import re
import numpy as np
# lets import some stuff
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import gensim
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

from gensim.models.keyedvectors import KeyedVectors

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
def load_google_news():
    embeddings_index = {}
    embeddings_index = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    return embeddings_index


def clean_str(string):
    """
    Tokenization/string cleaning for datasets.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels():
    """
    Loads polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(".././data/rt-polarity.pos", "r", encoding='latin-1').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(".././data/rt-polarity.neg", "r", encoding='latin-1').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    x_text = [s.split(" ") for s in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

def tokenizer_data (data) :
    max_features = 20000  # this is the number of words we care about
    sequence_length = 56;
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data)

    # this takes our sentences and replaces each word with an integer
    X = tokenizer.texts_to_sequences(data)

    # we then pad the sequences so they're all the same length (sequence_length)

    X = pad_sequences(X, maxlen= sequence_length)
    word_index = tokenizer.word_index
    # y = pd.get_dummies(data['Sentiment']).values

    # where there isn't a test set, Kim keeps back 10% of the data for testing, I'm going to do the same since we have an ok amount to play with
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    # print("test set size " + str(len(X_test)))
    return X, tokenizer

def create_pretrain_vectors (X):

    embeddings_index = load_google_news()
    X_train, tokenizer = tokenizer_data(X)
    num_words = len(tokenizer.word_index)
    print(num_words)
    embedding_dim = 300
    # first create a matrix of zeros, this is our embedding matrix
    embedding_matrix = np.random.uniform(size = (num_words+1, embedding_dim));
    for word, i in tokenizer.word_index.items():
        # print(word, i);
        # if(i > num_words): continue
        try:
            embedding_vector = embeddings_index[word]
            # print(embedding_vector)
            embedding_matrix[i] = embedding_vector
        except KeyError:
            embedding_matrix[i] = np.random.uniform(-0.25,0.25,embedding_dim)
    print(embedding_matrix[0])
    return X_train, embedding_matrix

def load_pre_train_data():
    x, y = load_data_and_labels();
    # X_train, word_index = tokenizer_data(x)
    X_train, embedding_matrix = create_pretrain_vectors(x)
    return [X_train,y, embedding_matrix]


def load_one_hot_vector():
    x,y = load_data_and_labels()
    X_train, tokenizer = tokenizer_data(x)
    return [X_train, y]

def save_embedding_layer(X):
    X_train, embedding_matrix = create_pretrain_vectors(X)

    file = open('embedding_matrix.txt', 'w')
    for line in embedding_matrix:
        file.write(' '.join([str(a) for a in line]) + '\n')

def load_embedding_vector():
    file = open('embedding_matrix.txt', 'r')
    embedding_layer = {}

    embedding_layer_file = list(open('embedding_matrix.txt', "r").readlines())
    embedding_layer_matrix = [s.strip() for s in embedding_layer_file]
    embedding_layer = [s.split(" ") for s in embedding_layer_matrix]
    return np.array(embedding_layer)

if __name__ == '__main__':
    x, y = load_data_and_labels();
    # # X_train, word_index = tokenizer_data(x)
    # X_train, embedding_matrix = create_pretrain_vectors(x)
    # print(X_train[0])
    # X_train, y = load_one_hot_vector()
    # print(y[0])
    # save_embedding_layer(x)
    print(load_embedding_vector())