import csv
import math

from fuzzywuzzy import fuzz
from laserembeddings import Laser
from scipy.spatial import distance
from sentence_transformers import SentenceTransformer
from nltk.tokenize import word_tokenize

laser = Laser()
sbert = SentenceTransformer('bert-base-nli-mean-tokens')


def get_laser_embedding(text, lang):
    if isinstance(text, list) or isinstance(text, tuple):
        return laser.embed_sentences(text, lang=lang)
    else:
        return laser.embed_sentences([text], lang=lang)


def get_sbert_embedding(text):
    if isinstance(text, list) or isinstance(text, tuple):
        return sbert.encode(text)
    else:
        return sbert.encode([text])


def angdist(u, v):
    return 1 - math.acos(1 - distance.cosine(u, v)) / math.pi


def cosine(u, v):
    return 1 - distance.cosine(u, v)


def generate_performance_report_file(counts, filename):
    report_str = 'score,laser,fuzzy,average,min\n'
    for i in range(100):
        report_str += '{},{},{},{},{}\n'.format(i+1, counts['laser'][i], counts['fuzzy'][i], counts['average'][i],
                                                counts['min'][i])

    with open("{}.csv".format(filename), "w") as report_file:
        report_file.write(report_str)


def generate_methods_report_files(pairs, filename):
    report_str = 'LookUp Item,Returned Item,LookUp Item Char Len,LookUp Item Word Len,FuzzyWuzzy Score,LASER Score,SBERT Score,Label\n'
    for pair in pairs:
        report_str += '{},{},{},{},{},{},{},{}\n'.format(pair['Database-Stored Sentence'], pair['Top Yielded Sentence'],
                                                         pair['LookUp Item Char Len'], pair['LookUp Item Word Len'],
                                                         pair['FuzzyWuzzy Score'], pair['LASER Score'],
                                                         pair['SBERT Score'], pair['Match class'])

    with open("{}.csv".format(filename), "w") as report_file:
        report_file.write(report_str)


def load_test_data():
    with open('../data/claim_pairs/laser_sample.csv') as csvfile:
        pairs = csv.reader(csvfile)
        pairs = [item for item in pairs]
    csv_headers = pairs[0]
    pairs = pairs[1:]

    with open('../data/claim_pairs/fuzzy_sample.csv') as csvfile:
        pairs_csv = csv.reader(csvfile)
        pairs += [item for item in pairs_csv][1:]

    with open('../data/claim_pairs/sbert_sample.csv') as csvfile:
        pairs_csv = csv.reader(csvfile)
        pairs += [item for item in pairs_csv][1:]

    with open('../data/claim_pairs/native_es_sample.csv') as csvfile:
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


def generate_density_plots_per_label():
    pairs = load_test_data()
    labels = [pair['Match class'] for pair in pairs]
    pairs = [(pair['Database-Stored Sentence'], pair['Top Yielded Sentence']) for pair in pairs]
    embedding_pairs = []
    for pair in pairs:
        embeddings = get_laser_embedding(pair, 'en')
        embedding_pairs.append((embeddings[0], embeddings[1]))
    laser_scores = [angdist(pair[0], pair[1]) * 100 for pair in embedding_pairs]
    fuzzy_scores = [fuzz.partial_ratio(pair[0], pair[1]) for pair in pairs]
    average_scores = [(laser_scores[i] + fuzzy_scores[i]) / 2 for i in range(len(pairs))]
    min_scores = [min(laser_scores[i], fuzzy_scores[i]) for i in range(len(pairs))]
    count = {
        'Very similar': {'laser': [0] * 100, 'fuzzy': [0] * 100, 'average': [0] * 100, 'min': [0] * 100},
        'Somewhat similar': {'laser': [0] * 100, 'fuzzy': [0] * 100, 'average': [0] * 100, 'min': [0] * 100},
        'Not similar': {'laser': [0] * 100, 'fuzzy': [0] * 100, 'average': [0] * 100, 'min': [0] * 100}
    }
    for i, label in enumerate(labels):
        count[label]['laser'][int(round(laser_scores[i])) - 1] += 1
        count[label]['fuzzy'][int(round(fuzzy_scores[i])) - 1] += 1
        count[label]['average'][int(round(average_scores[i])) - 1] += 1
        count[label]['min'][int(round(min_scores[i])) - 1] += 1
    generate_performance_report_file(count['Very similar'], 'very_similar')
    generate_performance_report_file(count['Somewhat similar'], 'somewhat_similar')
    generate_performance_report_file(count['Not similar'], 'not_similar')


def generate_methods_spreadsheet():
    pairs = load_test_data()
    for pair in pairs:
        pair['FuzzyWuzzy Score'] = fuzz.partial_ratio(pair['Database-Stored Sentence'], pair['Top Yielded Sentence'])
        laser_embeddings = get_laser_embedding([pair['Database-Stored Sentence'], pair['Top Yielded Sentence']], 'en')
        pair['LASER Score'] = angdist(laser_embeddings[0], laser_embeddings[1])
        sbert_embeddings = get_sbert_embedding([pair['Database-Stored Sentence'], pair['Top Yielded Sentence']])
        pair['SBERT Score'] = angdist(sbert_embeddings[0], sbert_embeddings[1])
        pair['LookUp Item Char Len'] = len(pair['Database-Stored Sentence'])
        pair['LookUp Item Word Len'] = len(word_tokenize(pair['Database-Stored Sentence']))

    generate_methods_report_files(pairs, 'methods_comparison')


if __name__ == "__main__":
    generate_methods_spreadsheet()
