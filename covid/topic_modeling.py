import spacy

spacy.load('en')
from spacy.lang.en import English
import nltk
# nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
import csv
from gensim import corpora
import pickle
import gensim
import string
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
import math
from scipy.spatial import distance

parser = English()
punctuation_translator = str.maketrans('', '', string.punctuation)
unisent_multilingual = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")


def get_sentence_embedding(text):
    return unisent_multilingual([text]).numpy()


def angdist(u, v):
    return 1 - math.acos(1 - distance.cosine(u, v)) / math.pi


def get_stop_words():
    # nltk.download('stopwords')
    # en_stop = set(nltk.corpus.stopwords.words('english'))
    with open('../data/stopwords-en.txt') as fp:
        stopwords = fp.readlines()
    with open('../data/stopwords-hi.txt') as fp:
        stopwords += fp.readlines()
    with open('../data/stopwords-pt.txt') as fp:
        stopwords += fp.readlines()
    stopwords = [sw.strip() for sw in stopwords]
    stopwords = set(stopwords)
    return stopwords


def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens


def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma


def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)


def prepare_text_for_lda(text):
    text = text.translate(punctuation_translator).lower()
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 2]
    tokens = [token for token in tokens if token not in stopwords]
    for i, token in enumerate(tokens):
        if token in ['covid-19', 'covid', 'covid19', 'coronavirus', 'virus', 'corona', 'vírus', 'coronavírus']:
            tokens[i] = 'virus'
    tokens = [get_lemma(token) for token in tokens]
    return tokens


stopwords = get_stop_words()


def do_topic_modeling_per_partner():
    partners, tip_line_requests = load_covid_data()

    for partner in partners:
        requests_tokens = []
        for tip in tip_line_requests[partner]:
            tokens = prepare_text_for_lda(tip['text'])
            requests_tokens.append(tokens)

        dictionary = corpora.Dictionary(requests_tokens)
        corpus = [dictionary.doc2bow(text) for text in requests_tokens]

        pickle.dump(corpus, open('corpus_{}.pkl'.format(partner), 'wb'))
        dictionary.save('dictionary_{}.gensim'.format(partner))

        NUM_TOPICS = 5
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=NUM_TOPICS, id2word=dictionary, passes=100)
        ldamodel.save('model100_{}.gensim'.format(partner))
        topics = ldamodel.print_topics(num_words=5)

        print('Partner: {}'.format(partner))
        for topic in topics:
            print(topic)
        print('##########################################')


def load_covid_data():
    with open('covid.csv') as csvfile:
        tip_line_requests = csv.reader(csvfile)
        tip_line_requests = [item for item in tip_line_requests]
    csv_headers = tip_line_requests[0]
    tip_line_requests = tip_line_requests[1:]

    temp_tip_line_requests = []
    for row in tip_line_requests:
        item = {}
        for i, key in enumerate(csv_headers):
            item[key] = row[i]
        temp_tip_line_requests.append(item)
    tip_line_requests = temp_tip_line_requests

    for tip in tip_line_requests:
        tip['text'] = tip['media_text'] if tip['media_text'] != 'NA' and len(tip['media_text']) >= len(tip['media_title']) else tip['media_title']
    tip_line_requests = [tip for tip in tip_line_requests if tip['text'] != 'NA']

    partners = set([item['team_slug'] for item in tip_line_requests])
    temp_tip_line_requests = {}
    for partner in partners:
        temp_tip_line_requests[partner] = [item for item in tip_line_requests if item['team_slug'] == partner]
    tip_line_requests = temp_tip_line_requests
    return partners, tip_line_requests


def textrank(texts, damping_factor=0.8, similarity_threshold=0.8):
    texts_embeddings = [get_sentence_embedding(p) for p in texts]

    text_similarities = {}
    for i, text in enumerate(texts):
        similarities = {}
        for j, embedding in enumerate(texts_embeddings):
            if i != j:
                similarities[texts[j]] = angdist(embedding, texts_embeddings[i])

        text_similarities[text] = similarities

    # create text rank matrix, add edges between pieces that are more than X similar
    matrix = np.zeros((len(texts), len(texts)))
    for i, i_text in enumerate(texts):
        for j, j_text in enumerate(texts):
            if i != j and text_similarities[i_text][j_text] > similarity_threshold:
                matrix[i][j] = text_similarities[i_text][j_text]

    scaled_matrix = damping_factor * matrix + (1 - damping_factor) / len(matrix)

    for row in scaled_matrix:
        row /= np.sum(row)
    # scaled_matrix = rescale(scaled_matrix)

    print('Calculating ranks...')
    ranks = np.ones((len(matrix), 1)) / len(matrix)
    iterations = 40
    for i in range(iterations):
        ranks = scaled_matrix.T.dot(ranks)

    return ranks


def extract_top_k_requests_per_topic(k, partner):
    ldamodel = gensim.models.ldamodel.LdaModel.load('model100_{}.gensim'.format(partner))
    with open('corpus_{}.pkl'.format(partner), 'rb') as corpus_file:
        corpus = pickle.load(corpus_file)
    # dictionary = corpora.Dictionary.load('dictionary_{}.gensim'.format(partner))

    topic_set = [[] for i in range(5)]
    for i, tip in enumerate(corpus):
        topics = ldamodel[tip]
        best_topic_id = 0
        best_topic_score = 0
        for topic in topics:
            if topic[1] > best_topic_score:
                best_topic_id = topic[0]
                best_topic_score = topic[1]
        topic_set[best_topic_id].append(i)

    _, tips = load_covid_data()
    partner_tips = tips[partner]

    results_per_topic = [[] for i in range(5)]
    for i, topic_ids in enumerate(topic_set):
        topic_tips = [tip['text'] for i, tip in enumerate(partner_tips) if i in topic_ids]
        ranks = textrank(topic_tips)
        results_per_topic[i] = [text for text, rank in sorted(zip(topic_tips, ranks), key=lambda item: item[1], reverse=True)[:min(k, len(topic_ids))]]

    return results_per_topic


if __name__ == "__main__":
    # do_topic_modeling_per_partner()

    partners = ['afp-fact-check', 'afp-checamos', 'india-today', 'boom-factcheck', 'africa-check']

    for partner in partners:
        ldamodel = gensim.models.ldamodel.LdaModel.load('model100_{}.gensim'.format(partner))
        topics = ldamodel.print_topics(num_words=5)

        report_str = ''

        report_str += 'Partner: {}\n'.format(partner)

        report_str = 'Top 5 Keywords Per Topic:\n'
        for topic in topics:
            report_str += str(topic)
        report_str += '##########################################\n'

        results_per_topic = extract_top_k_requests_per_topic(5, partner)
        for i, result_set in enumerate(results_per_topic):
            report_str += 'Examples for Topic {}:\n'.format(i)
            for result in result_set:
                report_str += result + '\n'
                report_str += '------------------------------------------\n'
            report_str += '##########################################\n'

        with open("report_{}.txt".format(partner), "w") as report_file:
            report_file.write(report_str)