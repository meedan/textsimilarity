import numpy as np
import json
import random
from utils import convert_from_hindi_latin, get_laser_embedding, vcosine, get_sbert_embedding, get_fuzzy_similarity_score


def load_samples(path):
    with open(path) as f:
        samples = json.load(f)
    return samples


def group_samples_by_language(samples):
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
    samples = load_samples('textsimilarity_samples_es.json')
    samples_per_language = group_samples_by_language(samples)
    # samples_per_language = downsample(samples_per_language, 1000)

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
        sample_texts = [item['text'] if language != 'hi-Latn' else item['transliterated_text'] for item in samples_per_language[language]]
        embeddings = get_sbert_embedding(sample_texts, language if language != 'hi-Latn' else 'hi')

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
    samples = load_samples('textsimilarity_samples_es.json')
    samples_per_language = group_samples_by_language(samples)

    pairs_to_annotate = []
    for language in samples_per_language:
        # laser_matrix = np.load('matrices/matrix_laser_{}.npy'.format(language))
        sbert_matrix = np.load('matrices/matrix_sbert_{}.npy'.format(language))
        # fuzzy_matrix = np.load('matrices/matrix_fuzzy_{}.npy'.format(language))

        index_pairs_to_annotate = set()
        # _select_indices_within_range(index_pairs_to_annotate, laser_matrix, 0.75, 0.9)
        # print('{} pairs in annotation set for lang={} with LASER'.format(len(index_pairs_to_annotate), language))
        _select_indices_within_range(index_pairs_to_annotate, sbert_matrix, 0.7, 0.9)
        print('{} pairs in annotation set for lang={} with SBERT'.format(len(index_pairs_to_annotate), language))
        # _select_indices_within_range(index_pairs_to_annotate, fuzzy_matrix, 0.7, 0.9)

        # removing cases with high overlap
        pairs_to_be_removed_below = []
        pairs_to_be_removed_above = []
        for (i, j) in index_pairs_to_annotate:
            fuzzy_score = get_fuzzy_similarity_score(samples_per_language[language][i], samples_per_language[language][j])
            if fuzzy_score <= 0.2:
                pairs_to_be_removed_below.append((i, j))
            elif fuzzy_score >= 0.9:
                pairs_to_be_removed_above.append((i, j))

        print('{} pairs to be removed with fuzzy similarity below 0.2'.format(len(pairs_to_be_removed_below)))
        print('{} pairs to be removed with fuzzy similarity above 0.9'.format(len(pairs_to_be_removed_above)))
        for (i, j) in pairs_to_be_removed_above + pairs_to_be_removed_below:
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


def evaluate_selected_pairs():
    samples = load_samples('textsimilarity_samples_es.json')
    samples_per_language = group_samples_by_language(samples)
    for language in samples_per_language:
        matrix = np.load('matrices/matrix_sbert_{}.npy'.format(language))
        over_90 = (matrix >= 0.9).sum() // 2
        range_80_90 = ((matrix >= 0.8) & (matrix < 0.9)).sum() // 2
        range_70_80 = ((matrix >= 0.7) & (matrix < 0.8)).sum() // 2

        print('lang={}, over 90: {}, between 80 to 90: {}, between 70 to 80: {}'.format(language, over_90, range_80_90, range_70_80))

        examples = []
        for i in range(len(matrix)):
            for j in range(i+1, len(matrix)):
                if matrix[i, j] >= 0.9:
                    examples.append([samples_per_language[language][i], samples_per_language[language][j]])

        examples = random.sample(examples, 20)
        for example in examples:
            print(example[0])
            print(example[1])
            print('-------------------------------')

        print('---------------------------------------------------------------------------------------------')


if __name__ == "__main__":
    generate_similarity_matrices()
    select_pairs_for_annotation()
    evaluate_selected_pairs()
