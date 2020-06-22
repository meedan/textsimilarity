import json
import math

from laserembeddings import Laser
from scipy.spatial import distance
from sentence_transformers import SentenceTransformer
from sklearn.metrics import confusion_matrix

laser = Laser()
sbert = SentenceTransformer('models/hindi-sxlmr-stmodel')


def angdist(u, v):
    return 1 - math.acos(cosine(u, v)) / math.pi


def cosine(u, v):
    return 1 - abs(distance.cosine(u, v))


def load_data():
    with open('../data/multilingual_sentence_matched_datasets.json') as f:
        sentence_pairs = json.load(f)
    return sentence_pairs['hindi_headlines']


def get_sbert_embedding(text):
    if isinstance(text, list) or isinstance(text, tuple):
        return sbert.encode(text)
    else:
        return sbert.encode([text])


def get_laser_embedding(text):
    if isinstance(text, list) or isinstance(text, tuple):
        return laser.embed_sentences(text, lang='hi')
    else:
        return laser.embed_sentences([text], lang='hi')


def main():
    pairs = load_data()
    labels = [pair['label'] for pair in pairs]

    laser_results = []
    sbert_results = []
    for pair in pairs:
        pair['lookup_text_laser_embedding'] = get_laser_embedding(pair['lookup_text'])
        pair['lookup_text_sbert_embedding'] = get_sbert_embedding(pair['lookup_text'])
        pair['database_text_laser_embedding'] = get_laser_embedding(pair['database_text'])
        pair['database_text_sbert_embedding'] = get_sbert_embedding(pair['database_text'])
    for threshold in range(1, 101):
        laser_predictions = [1 if angdist(pair['lookup_text_laser_embedding'],
                                          pair['database_text_laser_embedding']) > threshold / 100 else 0 for pair in
                             pairs]
        sbert_predictions = [1 if angdist(pair['lookup_text_sbert_embedding'],
                                          pair['database_text_sbert_embedding']) > threshold / 100 else 0 for pair in
                             pairs]
        tn, fp, fn, tp = confusion_matrix(labels, laser_predictions).ravel()
        laser_results.append([tn, fp, fn, tp])
        tn, fp, fn, tp = confusion_matrix(labels, sbert_predictions).ravel()
        sbert_results.append([tn, fp, fn, tp])
    print('threshold,laser_tn,laser_fp,laser_fn,laser_tp,sbert_tn,sbert_fp,sbert_fn,sbert_tp')
    for i in range(1, 101):
        print('{},{},{},{},{},{},{},{},{}'.format(i, laser_results[i][0], laser_results[i][1], laser_results[i][2],
                                                  laser_results[i][3], sbert_results[i][0], sbert_results[i][1],
                                                  sbert_results[i][2], sbert_results[i][3]))


if __name__ == "__main__":
    main()
