import csv
import math

from fuzzywuzzy import fuzz
from laserembeddings import Laser
from scipy.spatial import distance
from sklearn.metrics import confusion_matrix

laser = Laser()


def get_sentence_embedding(text, lang):
    if isinstance(text, list) or isinstance(text, tuple):
        return laser.embed_sentences(text, lang=lang)
    else:
        return laser.embed_sentences([text], lang=lang)


def angdist(u, v):
    return 1 - math.acos(1 - distance.cosine(u, v)) / math.pi


def cosine(u, v):
    return distance.cosine(u, v)


def generate_performance_report_file(counts, filename):
    report_str = 'laser,fuzzy,averae,min\n'
    for i in range(100):
        report_str += '{},{},{},{},{}\n'.format(i, counts['laser'][i], counts['fuzzy'][i], counts['average'][i], counts['min'][i])

    with open("{}.csv".format(filename), "w") as report_file:
        report_file.write(report_str)


def load_test_data():
    with open('data/claim_pairs/laser_sample.csv') as csvfile:
        pairs = csv.reader(csvfile)
        pairs = [item for item in pairs]
    csv_headers = pairs[0]
    pairs = pairs[1:]

    with open('data/claim_pairs/fuzzy_sample.csv') as csvfile:
        pairs_csv = csv.reader(csvfile)
        pairs += [item for item in pairs_csv][1:]

    temp_pairs = []
    for row in pairs:
        item = {}
        for i, key in enumerate(csv_headers):
            item[key] = row[i]
        temp_pairs.append(item)
    pairs = temp_pairs

    return pairs


pairs = load_test_data()
labels = [pair['Match class'] for pair in pairs]
pairs = [(pair['Database-Stored Sentence'], pair['Top Yielded Sentence']) for pair in pairs]

embedding_pairs = []
for pair in pairs:
    embeddings = get_sentence_embedding(pair, 'en')
    embedding_pairs.append((embeddings[0], embeddings[1]))

laser_scores = [cosine(pair[0], pair[1]) * 100 for pair in embedding_pairs]
fuzzy_scores = [fuzz.partial_ratio(pair[0], pair[1]) for pair in pairs]
average_scores = [(laser_scores[i] + fuzzy_scores[i]) / 2 for i in range(len(pairs))]
min_scores = [min(laser_scores[i] + fuzzy_scores[i]) for i in range(len(pairs))]

count = {
    'Very similar': {'laser': [0] * 100, 'fuzzy': [0] * 100, 'average': [0] * 100, 'min': [0] * 100},
    'Somewhat similar': {'laser': [0] * 100, 'fuzzy': [0] * 100, 'average': [0] * 100, 'min': [0] * 100},
    'Not similar': {'laser': [0] * 100, 'fuzzy': [0] * 100, 'average': [0] * 100, 'min': [0] * 100}
}

for i, label in enumerate(labels):
    count[label]['laser'][round(laser_scores[i])] += 1
    count[label]['fuzzy'][round(fuzzy_scores[i])] += 1
    count[label]['average'][round(average_scores[i])] += 1
    count[label]['min'][round(min_scores[i])] += 1

generate_performance_report_file(count['Very similar'], 'very_similar')
generate_performance_report_file(count['Somewhat similar'], 'somewhat_similar')
generate_performance_report_file(count['Not similar'], 'not_similar')
