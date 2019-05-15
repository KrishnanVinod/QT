import nltk as nlp
import string
import re
import tag
from collections import Counter
import pickle

tagger = tag.TrainTagger()
remove_punc = string.maketrans(string.punctuation, ' ' * len(string.punctuation))

class Officer:
    def __init__(self):
        return None

    def identify_gangs(self, sentences):
        # gangs are the categories you're looking for
        # e.g. products, colors, brands, etc
        # this uses the tagger to figure out
        # what type of gang this might belong to
        # using examples
        # e.g. [rug, table, chair] belong to [P] <- product
        to_return = list()
        for sentence in sentences:
            sentence = clean_up(sentence)
            for gram in nlp.ngrams(sentence, 2):
                tmp_gram = list(gram)
                for i,word in enumerate(tmp_gram):
                    tmp_gram[i] = tagger.tag_word(word)
                gram = tuple(tmp_gram)
                to_return.append(gram)
        return Counter(to_return)

    def find_associates(self, gangs, member):
        # returns common associates in different
        # gangs for a given member
        to_return = []
        to_hold = []
        for key in gangs.keys():
            if member in key[0]:
                to_hold.append([key[1],gangs.get(key)])
        to_hold = sorted(to_hold, key=lambda item: item[1], reverse = True)
        for line in to_hold:
            to_return.append({line[0]: line[1]})
        return to_return

def clean_up(sentence):
    sentence = sentence.translate(remove_punc)
    sentence = re.sub(' +', ' ', sentence)
    sentence = re.sub(r'[^\x00-\x7F]+', '', sentence)
    sentence = sentence.strip()
    sentence = nlp.tokenize.word_tokenize(sentence)
    return sentence
