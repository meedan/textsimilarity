from laserembeddings import Laser
from scipy.spatial import distance
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import numpy as np
import json
import random
from cleaning import convert_from_hindi_latin


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
    languages = set([item['language'] for item in samples])
    samples_per_language = {}
    for language in languages:
        samples_per_language[language] = [item for item in samples if item['language'] == language]

    for sample in samples_per_language['hi-Latn']:
        sample['transliterated_text'] = convert_from_hindi_latin(sample['text'])

    return samples_per_language


def generate_similarity_matrices():
    samples = load_samples('textsimilarity_samples.json')
    samples = random.sample(samples, 100)
    samples_per_language = group_samples_by_language(samples)

    for language in samples_per_language:
        # retrieving laser embeddings
        print('retrieving laser embeddings for language: {}'.format(language))
        sample_texts = [item['text'] for item in samples_per_language[language]]
        embeddings = get_laser_embedding(sample_texts, language)

        # generating laser matrices
        print('calculating laser similarity matrix for language: {}'.format(language))
        similarity_matrix = vcosine(embeddings, embeddings)
        np.fill_diagonal(similarity_matrix, 0)
        np.save('matrix_laser_{}'.format(language), similarity_matrix)

        # retrieving sbert embeddings
        print('retrieving sbert embeddings for language: {}'.format(language))
        sbert_model = get_sbert_model(language if language != 'hi-Latn' else 'hi')
        sample_texts = [item['text'] if language != 'hi-Latn' else item['transliterated_text'] for item in samples_per_language[language]]
        embeddings = get_sbert_embedding(sbert_model, sample_texts)

        # generating sbert matrices
        print('calculating sbert similarity matrix for language: {}'.format(language))
        similarity_matrix = vcosine(embeddings, embeddings)
        np.fill_diagonal(similarity_matrix, 0)
        np.save('matrix_sbert_{}'.format(language), similarity_matrix)

    for language in samples_per_language:
        sample_size = len(samples_per_language[language])
        similarity_matrix = np.zeros((sample_size, sample_size))

        print('calculating fuzzy similarity matrix for language: {}'.format(language))
        for i, sample in enumerate(samples_per_language[language]):
            for j, other_sample in enumerate(samples_per_language[language]):
                if i != j:
                    similarity_matrix[i, j] = get_fuzzy_similarity_score(sample['text'], other_sample['text'])

        np.save('matrix_fuzzy_{}'.format(language), similarity_matrix)


if __name__ == "__main__":
    generate_similarity_matrices()
