#! /bin/python3
#
"""
This script will compare 5 approaches to semantic text similarity using the 2017  STS SemEval data - http://ixa2.si.ehu.es/stswiki/index.php/Main_Page
Each sentence pair is rated 0-5. 0 indicates completely different, while 5 indicates identifical (see https://www.aclweb.org/anthology/S17-2001.pdf)

To evaluate these approaches, we take two subsets of the data
SAME subset - statement pairs with a score >= 4
DIFF subset - statement pairs with a score <=2

We calculate and want to maximise 
Normalised intermean distance: mean(f(SAME))-mean(f(DIFF))
Normalised intergroup distance: min(f(SAME))-max(f(DIFF))
By this notation, I mean that we apply some approch f to every sentence pair in SAME (or DIFF) resulting in a distribution of output scores
We then noramlise these scores and take the mean/max/min of the normalised scores


The functions we will compare are:
1. Mean word embedding and cosine distance 
	Embed each word in a statement and take the mean of these embeddings. 
	Compute the cosine distance between the mean of two statements. 
	This is the current approach in Alegre

2. Point cloud distance
	Embed each word in a statement, but do not average them!
	Instead compute the cosine distance between all cross-statement word pairs (cartesian product)
	For each word in statement 1, select its minimum distance to a word in statement 2. Sum these distances

3. Word mover distance
	https://radimrehurek.com/gensim/auto_examples/tutorials/run_wmd.html

4. CR5 document embeddings
	Embed each statement using CR5 and compute the cosine distance

5. Set intersection over set union
	No word embeddings here. Just plain bag-of-words appoach with no preprocessing
"""

import logging

import spacy
import gensim.downloader as api
from statistics import mean
from functools import partial
from scipy.spatial import distance
import numpy as np

def load_data(file):
	"""Load data from input file
	Return two lists of sentence pairs. The first is matching sentences and the second is non-matching sentences
	Parameters
		@file input file name
		@return SAME,DIFF . Two lists of lists. Each element of a list is a pair of sentences
	"""	 
	SAME=[]
	DIFF=[]
	with open(file, "r") as fh:
		for line in fh:
			stmt1,stmt2,score=line.strip().split("\t")
			try:
				score=float(score)
				if score>=4:
					SAME.append([stmt1,stmt2])
				elif score<=2:
					DIFF.append([stmt1,stmt2])
			except:
				print("ERROR with line {}".format(line))
	return SAME,DIFF

def preprocess(txt,parser):
	"""
	tokenize and lemmatize all words using spaCy
	"""
	doc = parser(txt)
	return [token.lemma_.lower() for token in doc]


def score(same_scores,diff_scores,inverse=False):
	"""
	Compute mean differences and max/min difference.
	Parameters:
		@same_scores - a list of scores, not normalised, for matching sentences
		@diff_scores - a list of scores, not normalised, for non-matching sentences
		@return intermean_dist,intergroup_dist - See comments at top of file
	"""
	norms=normalise(same_scores+diff_scores,inverse)
	s1=norms[0:len(same_scores)]
	s2=norms[len(same_scores):]
	intermean_dist=mean(s1)-mean(s2)
	intergroup_dist=min(s1)-max(s2)
	return intermean_dist,intergroup_dist

def normalise(scores,inverse=False):
	mn=min(scores)
	mx=max(scores)
	if inverse:
		return [1-((x-mn)/(mx-mn)) for x in scores]
	else:
		return [(x-mn)/(mx-mn) for x in scores]

def run_experiment(same, diff, prep_fun, dist_fun,inverse=False):
	same_dists=_run_experiment_set(same,prep_fun,dist_fun)
	diff_dists=_run_experiment_set(diff,prep_fun,dist_fun)
	return score(same_dists,diff_dists,inverse)

def _run_experiment_set(stmts, prep_fun, dist_fun):
	dists=[]
	for pair in stmts:
		dists.append(
				dist_fun(
						prep_fun(pair[0]),
						prep_fun(pair[1])
				)
			)
	return dists

#
# Word Mover Distance
#
def wordmover(stmt1,stmt2,model):
	return model.wmdistance(stmt1,stmt2)

#
# Mean word embedding and cosine distance  (current approach)
#
def run_mean_word_cos(same,diff,stopwords,model):
	#stmt1=[model.wv[w] for w in stmt1 if w in model]
	#stmt2=[model.wv[w] for w in stmt2 if w in model]
	#cos(mean(stmt1),mean(stmt2))
	from alegre_docsim import DocSim
	ds = DocSim(model,stopwords)
	return run_experiment(same,diff,ds.vectorize,lambda x,y : float(ds.cosine_sim(x,y)))
	#Note: for ds.cosine_sim, 1=most similar (0=dissimular)
	#lambda function is to force the return of a float rather than a numpy32 type variable (without changing the algrege_docsim.py file)

#
# Set approach
#
def set_dist(stmt1,stmt2):
	s1=set(stmt1)
	s2=set(stmt2)
	return len(s1.intersection(s2))/len(s1.union(s2))


#
# Cr5
#
def run_cr5(same,diff,preprocess_fun):
	# To use Cr5, you need cr5.py, which was downloaded from https://zenodo.org/record/2597441 in the same directory
	# You will also need to download some language embeddings from the same location and gunzip the files.
	# I saved the embeddings in a folder called models in ../Cr5/models

	from cr5 import Cr5_Model

	# path_to_pretrained_model, model_prefix
	cr5_model = Cr5_Model('../Cr5/models/','joint_28')

	cr5_model.load_langs(['en']) # list_of_languages being imported

	return run_experiment(same,diff,preprocess_fun,partial(cr5_dist,model=cr5_model),inverse=True)

def cr5_dist(stmt1,stmt2,model):
	s1_embedding = model.get_document_embedding(stmt1, 'en')
	s2_embedding = model.get_document_embedding(stmt2, 'en')

	return float(distance.cosine(s1_embedding, s2_embedding))

#
# Point cloud methods
#
def avg_sim_between_point_clouds(embeddings1, embeddings2):
	"""for all points in a point cloud (all token vectors in a caption),
		search the nearest point in the reference cloud
		and compute their (euclidean) distance,
		then take the average.
		
		This is similar to every iteration during Iterative Closest Point:
		https://en.wikipedia.org/wiki/Point_set_registration#Iterative_closest_point
		Besl, Paul; McKay, Neil (1992). "A Method for Registration of 3-D Shapes". IEEE Transactions on Pattern Analysis and Machine Intelligence. 14 (2): 239–256. doi:10.1109/34.121791.
	"""

    # Make cosine_scores_matrix
	tgt_emb = embeddings2   
	cosine_scores = np.array([ (tgt_emb / np.linalg.norm(tgt_emb, 2, 1)[:, None]).dot(word_emb / np.linalg.norm(word_emb))
		for word_emb in embeddings1 ])

	total_similarity = 0.0
	for row in cosine_scores:
		total_similarity += max(row).round(decimals=10)
	for col in cosine_scores.T:
		total_similarity += max(col).round(decimals=10)        

	n_scores = len(embeddings1)+len(embeddings2)
	
	return float(total_similarity/n_scores) #Inverted to be sim and not distance


def kernel_correlation_sim(embeddings1, embeddings2, bw=None):
	from sklearn.metrics.pairwise import rbf_kernel

	if bw is None:
		gamma = 1.0
	else:
		gamma = 1.0/bw
        
	similarity = rbf_kernel(embeddings1 ,embeddings2, gamma=gamma).sum()
	normalising  = 0.5*rbf_kernel(embeddings1, embeddings1, gamma=gamma).sum()
	normalising += 0.5*rbf_kernel(embeddings2, embeddings2, gamma=gamma).sum()
	return float(similarity/normalising) #Inverted to be sim and not dist

def wordcloud_embed(txt,parser,model):
	"""Tokenize txt, embed all words, and return a list of these embeddings (a list of lists)"""
	word_vecs=[]
	lemmas=preprocess(txt,parser)
	for l in lemmas:
		try:
			vec = model.wv[l]
			word_vecs.append(vec)
		except KeyError:
			# Ignore, if the word doesn't exist in the vocabulary
			pass
	return word_vecs


if __name__ == "__main__":
	logging.info("Loading...")
	#This is English-only and hence something to review in the future
	# Execute 
	# 	python -m spacy download en_core_web_sm
	# prior to loading this. See https://spacy.io/usage/models#quickstart
	ENparser = spacy.load("en_core_web_sm") 

	#We can use lots of embeddings. This is another axis for comparison.
	#We should definitely try MUSE and Bert
	# To get started, we'll use the standard word2vec model from Google News
	gn300model = api.load('word2vec-google-news-300')

	SAME,DIFF=load_data("./data/test_STS2017en-en.txt")

	logging.info("All initial setup complete")

	# Word mean + cosine
	print("Word vector means and cosine (current approach)...")
	stopwords_path = './data/stopwords-en.txt'
	with open(stopwords_path, 'r') as fh:
		stopwords = fh.read().split(',')
	x=run_mean_word_cos(SAME,DIFF,stopwords,gn300model)
	print(x)

	#Point cloud
	print("Word cloud average method...")
	x=run_experiment(SAME,DIFF,partial(wordcloud_embed,parser=ENparser,model=gn300model),avg_sim_between_point_clouds)
	print(x)

	#Point cloud kernal
	print("Word cloud kernal methods (third output is bandwidth)....")
	#gamma_range = np.arange(1,11,1)
	#bw_range    = 1.0/gamma_range[::-1]
	bw_range=[0.001,0.01,0.1,0.5,1.0]
	for bw in bw_range:
		#print("bw={}".format(bw))
		x=run_experiment(SAME,DIFF,partial(wordcloud_embed,parser=ENparser,model=gn300model),partial(kernel_correlation_sim,bw=bw))
		print(x+(bw,))

	# Word Mover Distance
	print("Word mover distance...")
	x=run_experiment(SAME,DIFF,partial(preprocess, parser=ENparser), partial(wordmover,model=gn300model),inverse=True)
	print(x)

	#CR5
	print("CR5 document embeddings")
	x=run_cr5(SAME,DIFF,partial(preprocess,parser=ENparser))
	print(x)

	#Set intersection over set union
	print("Set intersection over set union...")
	x=run_experiment(SAME,DIFF,partial(preprocess, parser=ENparser), set_dist)
	print(x)

	#TODO: Other embeddings (MUSE, Burt)

