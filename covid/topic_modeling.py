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
from laserembeddings import Laser
import math
import cld3
from scipy.spatial import distance
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.metrics import pairwise_distances_argmin_min


parser = English()
punctuation_translator = str.maketrans('', '', string.punctuation)
laser = Laser()

partner_languages = {
    'afp-checamos': ['pt'],
    'africa-check': ['en'],
    'afp-fact-check': ['en', 'hi'],
    'india-today': ['en', 'hi'],
    'boom-factcheck': ['en', 'hi']
}


def get_sentence_embedding(text, lang):
    if isinstance(text, list):
        return laser.embed_sentences(text, lang=lang)
    else:
        return laser.embed_sentences([text], lang=lang)


def angdist(u, v):
    return 1 - math.acos(1 - distance.cosine(u, v)) / math.pi


def cosine(u, v):
    return distance.cosine(u, v)


def get_stop_words():
    # nltk.download('stopwords')
    # en_stop = set(nltk.corpus.stopwords.words('english'))
    with open('../data/stopwords-en_2.txt') as fp:
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
        for language in tip_line_requests[partner]:
            requests_tokens = []
            for tip in tip_line_requests[partner][language]:
                tokens = prepare_text_for_lda(tip['text'])
                requests_tokens.append(tokens)

            dictionary = corpora.Dictionary(requests_tokens)
            corpus = [dictionary.doc2bow(text) for text in requests_tokens]

            pickle.dump(corpus, open('corpus_{}_{}.pkl'.format(partner, language), 'wb'))
            dictionary.save('dictionary_{}_{}.gensim'.format(partner, language))

            NUM_TOPICS = 5
            ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=NUM_TOPICS, id2word=dictionary, passes=100)
            ldamodel.save('model100_{}_{}.gensim'.format(partner, language))
            topics = ldamodel.print_topics(num_words=5)

            print('Partner: {}, Language: {}'.format(partner, language))
            for topic in topics:
                print(topic)
            print('##########################################')
    return partners, tip_line_requests


# embedding_cache = [{'key': None, 'value': None}, {'key': None, 'value': None}]
def is_a_match(a, b, threshold):
    # if a == embedding_cache[0]['key']:
    #     a_embedding = embedding_cache[0]['value']
    # else:
    #     a_embedding = get_sentence_embedding(a, lang)
    #     embedding_cache[0]['key'] = a
    #     embedding_cache[0]['value'] = a_embedding
    # if b == embedding_cache[1]['key']:
    #     b_embedding = embedding_cache[0]['value']
    # else:
    #     b_embedding = get_sentence_embedding(b, lang)
    #     embedding_cache[1]['key'] = b
    #     embedding_cache[1]['value'] = b_embedding
    return cosine(a, b) >= threshold


def remove_duplicates_based_on_pm_id(tips):
    pm_id_set = set([c['pm_id'] for c in tips])
    cleaned_tips = []
    for pm_id in pm_id_set:
        for tip in tips:
            if tip['pm_id'] == pm_id:
                cleaned_tips.append(tip)
                break

    return cleaned_tips


def remove_duplicate_requests(tips):
    checked_pm_ids = set()
    for tip in tips:
        if tip['pm_id'] in checked_pm_ids:
            continue
        for other_tip in tips:
            if tip != other_tip and is_a_match(tip['embedding'], other_tip['embedding'], 0.75):
                other_tip['pm_id'] = tip['pm_id']
                checked_pm_ids.add(tip['pm_id'])

    return remove_duplicates_based_on_pm_id(tips)


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
        lang_data = cld3.get_language(tip['text'])
        if lang_data is not None:
            tip['language'] = lang_data.language
    tip_line_requests = [tip for tip in tip_line_requests if tip['text'] != 'NA' and not tip['text'].isspace() and 'language' in tip]

    partners = set([item['team_slug'] for item in tip_line_requests])
    temp_tip_line_requests = {}
    for partner in partners:
        partner_tips = [item for item in tip_line_requests if item['team_slug'] == partner]
        temp_tip_line_requests[partner] = {lang: [] for lang in partner_languages[partner]}
        for tip in partner_tips:
            if tip['language'] in partner_languages[partner]:
                tip['embedding'] = get_sentence_embedding(tip['text'], tip['language'])
                temp_tip_line_requests[partner][tip['language']].append(tip)
        for language in partner_languages[partner]:
            temp_tip_line_requests[partner][language] = remove_duplicate_requests(temp_tip_line_requests[partner][language])

    tip_line_requests = temp_tip_line_requests
    return partners, tip_line_requests


def textrank(texts, texts_embeddings, damping_factor=0.8, similarity_threshold=0.8):
    text_similarities = {}
    for i, text in enumerate(texts):
        similarities = {}
        for j, embedding in enumerate(texts_embeddings):
            if i != j:
                similarities[texts[j]] = cosine(embedding, texts_embeddings[i])

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


def extract_top_k_requests_per_topic(k, partner, language, tips):
    ldamodel = gensim.models.ldamodel.LdaModel.load('model100_{}_{}.gensim'.format(partner, language))
    with open('corpus_{}_{}.pkl'.format(partner, language), 'rb') as corpus_file:
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

    if tips is None:
        _, tips = load_covid_data()
    partner_tips = tips[partner][language]

    results_per_topic = [[] for i in range(5)]
    for i, topic_ids in enumerate(topic_set):
        topic_tips= [tip['text'] for i, tip in enumerate(partner_tips) if i in topic_ids]
        topic_tips_embeddings = [tip['embedding'] for i, tip in enumerate(partner_tips) if i in topic_ids]
        ranks = textrank(topic_tips, topic_tips_embeddings)
        results_per_topic[i] = [text for text, rank in sorted(zip(topic_tips, ranks), key=lambda item: item[1], reverse=True)[:min(k, len(topic_ids))]]

    return results_per_topic


def generate_topic_modeling_report():
    global language
    print('starting topic modeling...')
    partners, tip_line_requests = do_topic_modeling_per_partner()
    print('topic modeling done.')
    # partners = ['afp-fact-check', 'afp-checamos', 'india-today', 'boom-factcheck', 'africa-check']
    for partner in partners:
        for language in partner_languages[partner]:
            ldamodel = gensim.models.ldamodel.LdaModel.load('model100_{}_{}.gensim'.format(partner, language))
            topics = ldamodel.print_topics(num_words=5)

            report_str = ''

            report_str += 'Partner: {}, Language: {}\n'.format(partner, language)

            report_str = 'Top 5 Keywords Per Topic:\n'
            for topic in topics:
                report_str += str(topic) + '\n'
            report_str += '##########################################\n'

            results_per_topic = extract_top_k_requests_per_topic(5, partner, language, tip_line_requests)
            for i, result_set in enumerate(results_per_topic):
                report_str += 'Examples for Topic {}:\n'.format(i)
                for result in result_set:
                    report_str += result + '\n'
                    report_str += '------------------------------------------\n'
                report_str += '##########################################\n'

            with open("report_{}_{}.txt".format(partner, language), "w") as report_file:
                report_file.write(report_str)


def get_sentences(text, lang):
    sentences = nltk.sent_tokenize(text, language=lang)
    sentences = [s for s in sentences if s and not s.isspace()]
    return sentences


if __name__ == "__main__":
    partners, tips = load_covid_data()

    # breaking each tip into sentences
    sentence_level_tips = {}
    for partner in tips:
        sentence_level_tips[partner] = {}
        for language in tips[partner]:
            sentence_level_tips[partner][language] = []
            for tip in tips[partner][language]:
                sentences = get_sentences(tip['text'], language)
                embeddings = get_sentence_embedding(sentences, language)
                sentence_level_tips[partner][language] += [{'text': sentences[i], 'embedding': embeddings[i]} for i in range(len(sentences))]

    # clustering sentences per partner per language
    for partner in sentence_level_tips:
        for language in sentence_level_tips[partner]:
            embeddings = [tip['embedding'] for tip in sentence_level_tips[partner][language]]
            n_clusters = max(5, round(len(embeddings)*0.0015))
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)

            # how many points per cluster
            predictions = kmeans.predict(embeddings)
            distribution = Counter(predictions)
            print(distribution.most_common(n_clusters))

            # closest points to centers
            closest_embeddings, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, embeddings)
            closest_sentences = []

            for closest_embedding in closest_embeddings:
                for tip in sentence_level_tips[partner][language]:
                    if tip['embedding'] == closest_embedding:
                        closest_sentences.append(tip['text'])
                        break
            print(closest_sentences)
            print('################################################')