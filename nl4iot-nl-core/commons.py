from nltk.tokenize import word_tokenize
import numpy as np
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
import fastText


def get_embeddings_model(configuration):
    embedding_engine = configuration.embedding_engine
    model_file = configuration.embeddings_model_file
    print('Start embeddings loading')
    if embedding_engine == 'Word2Vec':
        model = Word2Vec.load(model_file)
    elif embedding_engine == 'fastText':
        model = fastText.load_model(model_file)
    elif embedding_engine == 'KeyedVectors':
        model = KeyedVectors.load_word2vec_format(model_file)
    else:
        raise ValueError('Invalid embedding engine ' + embedding_engine)
    print('Finished embeddings loading')
    return model


def sentence2embeddings(sentence, max_length, model, embeddings_size):
    sentence = word_tokenize(sentence)
    return split_sentece2embeddings(sentence, max_length, model, embeddings_size)


def split_sentece2embeddings(sentence, max_length, model, embeddings_size):
    embeddings = np.zeros([max_length, embeddings_size])
    words_counter = 0
    for word in sentence:
        if word != " " and word != "":
            # word = word.lower()
            embeddings[words_counter] = get_word_embedding(word, model)
            words_counter += 1
            # if word in model.wv:
            #    embeddings[words_counter] = model.wv[word]
            #    words_counter += 1
            # elif word != " " and word != "":
            #    print(word, "not found in vocabulary")
    return embeddings, words_counter

def get_word_embedding(word, model):
    return model.get_word_vector(word).astype('float32')
