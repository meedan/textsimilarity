import csv
import json
import random

import cld3
import numpy as np
import pandas as pd
import requests
from cleaning import remove_emoji, spam_list

random_seed = 72
random.seed(random_seed)


def is_spam(text, lang):
    return len([item for item in spam_list[lang] if item in text]) > 0


def load_and_group_WA_data(path, group_sample_size=10000, languages=['en', 'hi', 'hi-Latn', 'mr', 'bn', 'ta', 'te', 'ml']):
    data = pd.read_csv(path, delimiter='\t', header=0)
    data = data.loc[data['language'].isin(languages)]
    data_per_language = {}
    for language in languages:
        df = data.loc[data['language'] == language]
        df = df.sample(min(group_sample_size, len(df)), random_state=random_seed)
        data_per_language[language] = df

    return data_per_language


def get_partner_languages():
    return {
        'afp-checamos': ['pt', 'en'],
        'africa-check': ['en'],
        'afp-fact-check': ['en', 'hi', 'hi-Latn', 'mr', 'bn'],
        'india-today': ['en', 'hi', 'hi-Latn', 'mr', 'bn'],
        'boom-factcheck': ['en', 'hi', 'hi-Latn', 'mr', 'bn']
    }


def load_tip_line_claim_data(path):
    with open(path) as csvfile:
        tip_line_requests = csv.reader(csvfile)
        tip_line_requests = [item for item in tip_line_requests]
    csv_headers = tip_line_requests[0]
    tip_line_requests = tip_line_requests[1:]
    # for test only
    # tip_line_requests = random.sample(tip_line_requests, 100)

    temp_tip_line_requests = []
    for row in tip_line_requests:
        item = {}
        for i, key in enumerate(csv_headers):
            item[key] = row[i]
        temp_tip_line_requests.append(item)
    return [item for item in temp_tip_line_requests if item['claim_type'] == 'Claim']


def group_tiplines_by_language(tip_line_requests, languages=['en', 'pt', 'hi', 'hi-Latn', 'mr', 'bn', 'ta', 'te', 'ml']):
    for tip in tip_line_requests:
        tip['text'] = remove_emoji(
            tip['media_text'] if tip['media_text'] != 'NA' and len(tip['media_text']) >= len(tip['media_title']) else
            tip['media_title'])
        lang_data = cld3.get_language(tip['text'])
        if lang_data is not None:
            tip['language'] = lang_data.language
    tip_line_requests = [tip for tip in tip_line_requests if tip['text'] != 'NA' and not tip['text'].isspace() and 'language' in tip and len(tip['text']) > 20]

    temp_tip_line_requests = {}
    for language in languages:
        temp_tip_line_requests[language] = [item for item in tip_line_requests if item['language'] == language]

    tip_line_requests = temp_tip_line_requests
    return tip_line_requests


def cosine_sim(vecA, vecB):
    """Find the cosine similarity distance between two vectors."""
    csim = np.dot(vecA, vecB) / (np.linalg.norm(vecA) * np.linalg.norm(vecB))
    if np.isnan(np.sum(csim)):
        return 0
    return csim


class AlegreClient:
    @staticmethod
    def default_hostname():
        return "http://0.0.0.0:5000"

    @staticmethod
    def text_similarity_path():
        return "/text/similarity/"

    def __init__(self, use_fuzzy=False, hostname=None):
        if not hostname:
            hostname = AlegreClient.default_hostname()
        self.hostname = hostname
        self.use_fuzzy = use_fuzzy

    def input_cases(self, texts, model_name, context={}, language=None):
        for text in texts:
            request_params = {"model": model_name, "text": text}
            if context:
                request_params["context"] = context
            if language:
                request_params["language"] = language
            if self.use_fuzzy:
                request_params["fuzzy"] = "auto"
            self.store_text_similarity(request_params)

    def get_for_text(self, text, model_name, context={}, language=None):
        if not context:
            context = {"task": "model_evaluation", "model_name": model_name}
        return json.loads(self.get_similar_texts({
            "model": model_name,
            "text": text.lower(),
            "context": context,
            "threshold": 0.0,
            "language": language
        }).text)

    def store_text_similarity(self, request_params):
        return requests.post(self.hostname + self.text_similarity_path(), json=request_params)

    def get_similar_texts(self, request_params):
        return requests.get(self.hostname + self.text_similarity_path(), json=request_params)


def sample_data_from_all_sources():
    group_sample_size = 10000
    tipline_data = load_tip_line_claim_data('../data/tiplines.csv')
    tipline_data = group_tiplines_by_language(tipline_data)
    public_groups_data = load_and_group_WA_data('../data/pg_text_spamfree.csv')
    languages = ['en', 'pt', 'hi', 'hi-Latn', 'mr', 'bn', 'ta', 'te', 'ml']
    samples = []
    id = 0
    for language in languages:
        if language not in public_groups_data:
            language_samples = random.sample(tipline_data[language],
                                             min(len(tipline_data[language]), group_sample_size))
            language_samples = [{'id': id + i, 'text': item['text'], 'language': language} for i, item in
                                enumerate(language_samples)]
            samples += language_samples
            id += len(language_samples)
        else:
            tipline_sample = random.sample(tipline_data[language],
                                           min(len(tipline_data[language]), group_sample_size // 2))
            pg_sample_size = group_sample_size // 2 if len(
                tipline_sample) == group_sample_size / 2 else group_sample_size - len(tipline_sample)
            pg_sample = public_groups_data[language].sample(pg_sample_size, random_state=random_seed)
            language_samples = [{'id': id + i, 'text': item['text'], 'language': language} for i, item in
                                enumerate(tipline_sample)]
            id += len(tipline_sample)
            language_samples += [{'id': id + i, 'text': item['message_text'], 'language': language} for i, item in
                                 pg_sample.iterrows()]
            id += len(pg_sample)
            samples += language_samples

    with open('textsimilarity_samples.json', 'w') as fp:
        json.dump(samples, fp)


if __name__ == '__main__':
    sample_data_from_all_sources()
