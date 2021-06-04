import re
import numpy as np
# lets import some stuff
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
def load_glove():
    embeddings_index = {}
    f = open("glove.6B.50d.txt", "r", encoding="utf8")
    #print(f.readlines())
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index;

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
    positive_examples = list(open("./data/rt-polarity.pos", "r", encoding='latin-1').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open("./data/rt-polarity.neg", "r", encoding='latin-1').readlines())
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

    embeddings_index = load_glove()
    X_train, tokenizer = tokenizer_data(X)
    num_words = len(tokenizer.word_index)
    print(num_words)
    embedding_dim = 50
    # first create a matrix of zeros, this is our embedding matrix
    embedding_matrix = np.zeros((num_words+1, embedding_dim));
    for word, i in tokenizer.word_index.items():
        print(word, i);
        # if(i > num_words): continue
        embedding_vector = embeddings_index.get(word)
        # print(embedding_vector)
        if embedding_vector is not None:
            # we found the word - add that words vector to the matrix
            for j in range(0,50,1):
                embedding_matrix[i][j] = embedding_vector[j]
        else:
            # doesn't exist, assign a random vector
            embedding_matrix[i] = np.random.randn(embedding_dim)
    return X_train, embedding_matrix

def load_pre_train_data():
    x, y = load_data_and_labels();
    # X_train, word_index = tokenizer_data(x)
    X_train, embedding_matrix = create_pretrain_vectors(x)
    return [X_train,y, embedding_matrix]

if __name__ == '__main__':
    x, y = load_data_and_labels();
    # X_train, word_index = tokenizer_data(x)
    X_train, embedding_matrix = create_pretrain_vectors(x)
    print(embedding_matrix[5])