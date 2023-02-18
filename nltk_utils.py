import nltk
#nltk.download('punkt')
import numpy as np

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bagset(tokenized_sentence,wordset):

    tokenized_sentence=[stem(w) for w in tokenized_sentence]

    bagset=np.zeros(len(wordset),dtype=np.float32)
    for idx, w in enumerate(wordset):
        if w in tokenized_sentence:
            bagset[idx]=1.0
    return bagset
