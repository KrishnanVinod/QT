import nltk as nlp
import string
import re
import numpy as np

remove_punc = str.maketrans(string.punctuation, ' '*len(string.punctuation))

#not sure what I was doing here

class Parser:
    def __init__(self):
        return None

    def chew(self, sentence):
        return clean_up(sentence)

    def swallow(self, morsels):
        for morsel in morsels:
            return morsel


    def digest(self, morsels):
        return None



def clean_up(sentence):
    sentence = sentence.translate(remove_punc)
    sentence = re.sub(' +', ' ', sentence)
    sentence = re.sub(r'[^\x00-\x7F]+', '', sentence)
    sentence = sentence.strip()
    sentence = nlp.tokenize.word_tokenize(sentence)
    return sentence
