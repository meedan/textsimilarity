from laserembeddings import Laser
from scipy.spatial import distance
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import numpy as np
import json
import random
from cleaning import convert_from_hindi_latin, remove_urls


laser = Laser()
indian_sbert = SentenceTransformer('../multilingual-sbert/models/se-asian-sbert')
portuguese_sbert = SentenceTransformer('distiluse-base-multilingual-cased')
english_sbert = SentenceTransformer('bert-base-nli-mean-tokens')


def get_sbert_model(language):
    if language == 'pt':
        return portuguese_sbert
    elif language in ['hi', 'ml', 'mr', 'ta', 'te', 'bn', 'hi-Latn']:
        return indian_sbert
    elif language == 'en':
        return english_sbert
    else:
        return None


def vcosine(u, v):
    return abs(1 - distance.cdist(u, v, 'cosine'))


def get_sbert_embedding(model, text):
    if isinstance(text, list) or isinstance(text, tuple):
        return model.encode(text)
    else:
        return model.encode([text])


def get_laser_embedding(text, lang):
    if isinstance(text, list) or isinstance(text, tuple):
        return laser.embed_sentences(text, lang=lang)
    else:
        return laser.embed_sentences([text], lang=lang)


def get_fuzzy_similarity_score(a, b):
    return fuzz.partial_ratio(a, b) / 100


def load_samples(path):
    with open(path) as f:
        samples = json.load(f)
    return samples


def group_samples_by_language(samples):
    for sample in samples:
        sample['text'] = remove_urls(sample['text'])

    languages = set([item['language'] for item in samples])
    samples_per_language = {}
    for language in languages:
        samples_per_language[language] = [item for item in samples if item['language'] == language]

    for sample in samples_per_language['hi-Latn']:
        sample['transliterated_text'] = convert_from_hindi_latin(sample['text'])

    return samples_per_language


def downsample(samples_per_language, downsample_size):
    random.seed(72)
    for language in samples_per_language:
        for sample in samples_per_language[language]:
            sample['random_id'] = random.uniform(0, 1)
        samples_per_language[language] = sorted(samples_per_language[language], key=lambda item: item['random_id'])
        samples_per_language[language] = samples_per_language[language][:downsample_size]

    return samples_per_language


def generate_similarity_matrices():
    samples = load_samples('textsimilarity_samples.json')
    samples_per_language = group_samples_by_language(samples)
    samples_per_language = downsample(samples_per_language, 1000)

    for language in samples_per_language:
        # retrieving laser embeddings
        print('retrieving laser embeddings for language: {}'.format(language))
        sample_texts = [item['text'] for item in samples_per_language[language]]
        embeddings = get_laser_embedding(sample_texts, language)

        # generating laser matrices
        print('calculating laser similarity matrix for language: {}'.format(language))
        similarity_matrix = vcosine(embeddings, embeddings)
        np.fill_diagonal(similarity_matrix, 0)
        np.save('matrices/matrix_laser_{}'.format(language), similarity_matrix)

        # retrieving sbert embeddings
        print('retrieving sbert embeddings for language: {}'.format(language))
        sbert_model = get_sbert_model(language if language != 'hi-Latn' else 'hi')
        sample_texts = [item['text'] if language != 'hi-Latn' else item['transliterated_text'] for item in samples_per_language[language]]
        embeddings = get_sbert_embedding(sbert_model, sample_texts)

        # generating sbert matrices
        print('calculating sbert similarity matrix for language: {}'.format(language))
        similarity_matrix = vcosine(embeddings, embeddings)
        np.fill_diagonal(similarity_matrix, 0)
        np.save('matrices/matrix_sbert_{}'.format(language), similarity_matrix)

    # for language in samples_per_language:
    #     sample_size = len(samples_per_language[language])
    #     similarity_matrix = np.zeros((sample_size, sample_size))
    #
    #     print('calculating fuzzy similarity matrix for language: {}'.format(language))
    #     for i, sample in enumerate(samples_per_language[language]):
    #         for j in range(i+1, len(samples_per_language[language])):
    #             other_sample = samples_per_language[language][j]
    #             similarity_matrix[i, j] = get_fuzzy_similarity_score(sample['text'], other_sample['text'])
    #
    #     np.save('matrices/matrix_fuzzy_{}'.format(language), similarity_matrix)


def select_pairs_for_annotation():
    samples = load_samples('textsimilarity_samples.json')
    samples_per_language = group_samples_by_language(samples)

    pairs_to_annotate = []
    for language in samples_per_language:
        laser_matrix = np.load('matrices/matrix_laser_{}.npy'.format(language))
        sbert_matrix = np.load('matrices/matrix_sbert_{}.npy'.format(language))
        # fuzzy_matrix = np.load('matrices/matrix_fuzzy_{}.npy'.format(language))

        index_pairs_to_annotate = set()
        _select_indices_within_range(index_pairs_to_annotate, laser_matrix, 0.75, 0.9)
        print('{} pairs in annotation set for lang={} with LASER'.format(len(index_pairs_to_annotate), language))
        _select_indices_within_range(index_pairs_to_annotate, sbert_matrix, 0.7, 0.9)
        print('{} pairs in annotation set for lang={} when adding SBERT'.format(len(index_pairs_to_annotate), language))
        # _select_indices_within_range(index_pairs_to_annotate, fuzzy_matrix, 0.7, 0.9)

        # removing cases with high overlap
        for (i, j) in index_pairs_to_annotate:
            if get_fuzzy_similarity_score(samples_per_language[language][i], samples_per_language[language][j]) >= 0.9:
                index_pairs_to_annotate.remove((i, j))

        print('{} pairs in annotation set for lang={} after cleaning up'.format(len(index_pairs_to_annotate), language))

        # find disputes between models and add them to annotation list
        # for i in range(len(laser_matrix)):
        #     for j in range(i+1, len(laser_matrix[i])):
        #         if abs(laser_matrix[i, j] - sbert_matrix[i, j]) >= 0.2:
        #             index_pairs_to_annotate.add((i, j))
        #         elif abs(laser_matrix[i, j] - fuzzy_matrix[i, j]) >= 0.3:
        #             index_pairs_to_annotate.add((i, j))
        #         elif abs(sbert_matrix[i, j] - fuzzy_matrix[i, j]) >= 0.3:
        #             index_pairs_to_annotate.add((i, j))

        pairs_to_annotate += [{'item1': samples_per_language[language][i], 'item2': samples_per_language[language][j], 'language': language} for (i, j) in index_pairs_to_annotate]

    with open('pairs_to_annotate.json', 'w') as fp:
        json.dump(pairs_to_annotate, fp)


def _select_indices_within_range(index_pairs_to_annotate, matrix, range_begin, range_end):
    for i, row in enumerate(matrix):
        for j in range(i + 1, len(row)):
            item = row[j]
            if range_begin <= item <= range_end:
                index_pairs_to_annotate.add((i, j))


if __name__ == "__main__":
    generate_similarity_matrices()
    select_pairs_for_annotation()
