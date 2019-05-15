from querytagger import tag
from collections import Counter

t = tag.TrainTagger()

stats = []


for f in open("queries_test.txt", 'r'):
    stats.append(t.tag_query(f))

