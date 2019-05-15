import nltk as nlp
import re
import string
import pickle
import itertools


remove_punc = str.maketrans(string.punctuation, ' ' * len(string.punctuation))

class Tagset:

    def __init__(self):
        self.tag_name = {'/B':'brand', '/Cn':'connector', '/M':'material', '/D':'dimensions', '/S':'shape', '/U':'unknown', '/P':'product', '/Cr':'color', '/G':'gender'}
        self.tag = {'brand':'/B', 'connector':'/Cn', 'material':'/M', 'dimensions':'/D', 'shape':'/S',
                         'unknown':'/U', 'product':'/P', 'color':'/Cr', 'gender':'/G'}

    def show_tags(self):
        return self.tags

    def translate_tag(self, tag):
        return self.tags.get(tag)



class TrainTagger:
    # with open ('trained_gangs', 'rb') as fp:
    #     n = pickle.load(fp)

    def __init__(self):

        self.colors = ['black', 'grey', 'gray', 'white', 'blue', 'green', 'red', 'orange', 'ivory', 'navy', 'brown',
                  'yellow', 'golden', 'teal', 'rainbow', 'multi']

        self.gender = ['his', 'her', 'men', 'mens', 'man', 'woman', 'womens', 'he', 'she']

        self.brands = ['24/7 comfort', 'isotoner', 'abbyson living', 'acer', 'adidas', 'aerosoles', 'akribos xxiv / twi watches',
                  'allied brass',
                  'amd', 'american apparel', 'anolon', 'apple', 'armani', 'armasight', 'ashanti', 'ashley furniture',
                  'asics', 'asus',
                  'atlantis', 'augason farms', 'august steiner', 'auriya', 'simran', 'balenciaga', 'banana republic',
                  'barbie',
                  'basacc', 'baxton studio', 'bcbg', 'bebe', 'belkin', 'bestar', 'betsey johnson', 'bissell',
                  'black pine sports',
                  'blanco', 'bosch', 'brooks brothers', 'brother', 'bugatti', 'bulova', 'burberry', 'burgi',
                  'twi watches', 'bushnell',
                  'cake boss', 'calvin klein', 'canon', 'cartier', 'casio', 'celestron', 'chanel', 'christopher knight',
                  'cisco',
                  'citizen', 'coach', 'coca cola', 'cole haan', 'coleman', 'columbia', 'converse', 'cricut',
                  'cuisinart', 'david yurman',
                  'dell', 'dg casa', 'dickies', 'diesel', 'dior', 'disney', 'dooney & bourke', 'dyson', 'echo',
                  'eddie bauer',
                  'eileen fisher', 'electrolux', 'excel', 'faberware', 'fendi', 'fisher price', 'flow wall', 'fossil',
                  'frigidaire',
                  'garmin', 'general electric', 'gigi hill', 'giuseppe zanotti', 'givenchy', 'gloria vanderbilt',
                  'go pro',
                  'gorilla playsets', 'gucci', 'guess', 'harley davidson', 'hello kitty', 'hermes', 'hilton',
                  'honeywell', 'hp', 'htc',
                  'hugo boss', 'ibm', 'intel', 'invicta', 'jaguar', 'jeep', 'jessica simpson', 'jimmy choo', 'jordan',
                  'juicy couture'
            , 'kate spade', 'keds', 'kenneth cole', 'keurig', 'kitchenaid', 'kohler', 'kraus', "l'oreal", 'lacoste',
                  'lancome', 'laura ashley'
            , 'laura ashley', 'lego', 'lenovo', 'lenox', 'lg', 'linksys', 'little tikes', 'logitech', 'longchamp',
                  'lucky brand', 'luminox',
                  'lush decor', 'madison park', 'marc jacobs', 'martha stewart', 'matrix', 'maui jim', 'mia', 'miadora',
                  'michael valitutti',
                  'micheal kors', 'michele', 'microsoft', 'mitsubishi', 'mizone', 'moen', 'motorola', 'movado',
                  'munchkin', 'nascar',
                  'nautica', 'nespresso', 'netgear', 'new balance', 'nike', 'nikon', 'nine west', 'ninja', 'nintendo',
                  'nixon', 'nokia', 'north face',
                  'nostalgia electrics', 'nuloom', 'nvidia', 'oakley', 'odyssey', 'omega', 'otterbox', 'panasonic',
                  'patagonia', 'paula deen',
                  'pioneer', 'plantronics', 'playstation', 'polaris', 'prada', 'puma', 'ralph lauren', 'rayban',
                  'razor', 'reebok', 'remington',
                  'rolex', 'roommates', 'safavieh', 'saint laurent', 'salvatore ferragamo', 'samsonite', 'samsung',
                  'saucony', 'seiko', 'serta',
                  'shark', 'sigma', 'silhouette', 'skechers', 'skullcandy', 'sony', 'sperry', 'steve madden',
                  'sweet jojo', 'tag heuer', 'taylormade',
                  'timberland', 'timex', 'tissot', 'tom ford', 'tommy bahama', 'tommy hilfiger', 'tomtom', 'toshiba',
                  'tribecca home', 'true religion',
                  'true religion jeans', 'turtle beach', 'ugg', 'under armour', 'vans', 'vera bradley', 'vera wang',
                  'versace', 'victorinox',
                  'vince camuto', 'vitamix', 'vizio', 'webkinz', 'whirlpool', 'yamaha']

        self.connectors = ['and', 'with', 'or', 'for', 'to', 'on', 'together', 'set', 'piece', 'sets', 'pieces', 'of', 'the']
        self.materials = ['diamond', 'gold', 'wood', 'wooden', 'steel', 'stainless', 'cotton', 'synthetic', 'down',
                     'alternative', 'chrome', 'iron', 'leather', 'plastic', 'handmade']
        self.shapes = ['round', 'square', 'long', 'short', 'circle', 'oval', 'area', 'slim', 'large', 'wide', 'tall',
                  'rectangle', 'rectangular']
        self.product = ['rug', 'rugs', 'runner', 'runners', 'shoes', 'shoe', 'table', 'chair', 'chairs', 'sofa', 'couch',
                   'sofas', 'tables',
                   'ottoman', 'pillow', 'comforter', 'comforters', 'duvets', 'duvet', 'bed', 'daybed', 'trundle']
        return None

    def tag_word(self, word):
        if word in self.colors:
            return word + "/Cr"
        if word in self.brands:
            return word + "/B"
        if word in self.connectors:
            return word + "/Cn"
        if word in self.materials:
            return word + "/M"
        for dim in ['x', 'by', 'small', 'big', 'large', 'medium', 'square', 'circle', 'oval', 'round']:
            if dim in word:
                return word + "/D"
            elif word.isdigit():
                return word + "/D"
        if word in self.shapes:
            return word + "/S"
        if word in self.product:
            return word + "/P"
        if word in self.gender:
            return word + '/G'
        return word + "/U"

    def tag_query(self, query):
        to_return = []
        for term in clean_up(query):
            to_return.append(self.tag_word(term))
        return to_return

    def train(self, queries):
        trained = []
        trained.append([])
        trained.append([])
        for query in queries:
            for word in clean_up(query):
                trained[query] = self.tag_word(word)


class Tagger:

    def __init__(self, trained_data_path):
        with open(trained_data_path, 'rb') as fp:
            self.n = pickle.load(fp)
        self.tag_name = {'/B':'brand', '/Cn':'connector', '/M':'material', '/D':'dimensions', '/S':'shape', '/U':'product', '/P':'product', '/Cr':'color'}
        self.tag = {'brand':'/B', 'connector':'/Cn', 'material':'/M', 'dimensions':'/D', 'shape':'/S',
                         'unknown':'/U', 'product':'/P', 'color':'/Cr'}
        self.legal_queries = [['P'], ['P', 'D'], ['D', 'P'], ['D', 'Cr', 'P'], ['Cr', 'P'], ['Cr', 'M'], ['M'], ['B'],
                              ['B', 'P'], ['Cr', 'P', 'P'], ['Cr', 'Cr', 'P'], ['M', 'Cr', 'P'], ['P', 'Cn', 'P'], ['P', 'Cn', 'M'], ['P', 'P', 'P'], ['D', 'D', 'P']]
        return None

    def tag_word(self, word, i=None):
        if i is None:
            i = 0
        pots = self._find_associates(self.n, word)
        return pots[i].keys()[0]

    def tag_query(self, query, i=None):
        to_return = []
        put_back = query.split(' ')
        for term in clean_up(query):
            to_return.append(self._find_associates(self.n, term))
        most_likely = []
        for a, b, c in itertools.combinations(to_return, 3):
           for i in a:
               for j in b:
                   for k in c:
                        print([put_back[0] + "/" + i.keys()[0], put_back[1] + "/" + j.keys()[0],
                               put_back[2] + "/" + k.keys()[0],
                               i.get(i.keys()[0]) + j.get(j.keys()[0]) + k.get(k.keys()[0])])


        # is_legal = self.check_legality(to_return)
        # if is_legal:
        #     return to_return
        # else:
        #     to_return = []
        #     for term in clean_up(query):
        #         to_return.append(self.tag_word(term, is_legal))

        return to_return

    def check_legality(self, types, j=None):
        if j is None:
            j = 0
        if not types in self.legal_queries:
            j += 1
            return j
        return True

    def _find_associates(self, grams, word):
        to_return = []
        to_hold = []
        for key in grams.keys():
            if word == key[0].split('/')[0]:
                if key[1].split('/')[1] is not 'U':
                    if key[1].split('/')[1] in [row[0] for row in to_hold]:
                        to_hold[[row[0] for row in to_hold].index(key[1].split('/')[1])][1] += grams.get(key)
                    else:
                        to_hold.append([key[1].split('/')[1],grams.get(key)])
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