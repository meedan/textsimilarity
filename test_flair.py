# https://github.com/flairNLP/flair/blob/master/resources/docs/embeddings/TRANSFORMER_EMBEDDINGS.md#xlmrobertaembeddings


"""
from flair.embeddings import XLMRobertaEmbeddings
from flair.data import Sentence

# init embedding
embedding = XLMRobertaEmbeddings()

# create a sentence
sentence = Sentence("J'aime le camembert, and best whishes from München und Berlin .")

# embed words in sentence
x=embedding.embed(sentence)
for token in sentence:
    #print(token)
    print(len(token.embedding))
"""

#Doesn't work
#y=sentence.get_embedding()
#print(y)
#print(type(y))
#print(len(y))

from flair.data import Sentence
from wmdistance import wmdistance
import numpy as np
    
#quit(0)

def mean_vec(v):
	return np.mean(v,axis=0)

def max_vec(v):
	return np.max(v,axis=0)

def flair_embed_matrix(txt,model):
	sentence = Sentence(txt)

	# embed words in sentence
	x=model.embed(sentence)
	return np.array([token.embedding.numpy() for token in sentence])
	
def flair_embed_dict(txt,model):
	sentence = Sentence(txt)

	# embed words in sentence
	x=model.embed(sentence)
	return {token.text:token.embedding.numpy() for token in sentence}


if __name__ == "__main__":

	from test_textsim import *
	from flair.embeddings import XLMRobertaEmbeddings, BertEmbeddings, XLNetEmbeddings, XLMEmbeddings, RoBERTaEmbeddings

	
	measures={}
	
	SAME,DIFF=load_data("./data/test_STS2017en-en.txt")
	
	MODELS={
		"xlmr":XLMRobertaEmbeddings(),
		"bert":BertEmbeddings(),
		"xlnet":XLNetEmbeddings(),
		"xlm":XLMEmbeddings(),
		"roberta":RoBERTaEmbeddings(),
	}
	
	for model in MODELS:	
	
		print(model)
		
		results = run_experiment(SAME,DIFF,lambda x: flair_embed_dict(x,MODELS[model]),	wmdistance,inverse=True)
		measures['{}-wmdist'.format(model)] = results
		print(score(results[0],results[1]))

		results = run_experiment(SAME,DIFF,lambda x: max_vec(flair_embed_matrix(x,MODELS[model])),distance.cosine,inverse=True)
		measures['{}-cosmax'.format(model)] = results
		print(score(results[0],results[1]))
	
		results = run_experiment(SAME,DIFF,lambda x: mean_vec(flair_embed_matrix(x,MODELS[model])),distance.cosine,inverse=True)
		measures['{}-cosine'.format(model)] = results
		print(score(results[0],results[1]))
	
		results = run_experiment(SAME,DIFF,lambda x: mean_vec(flair_embed_matrix(x,MODELS[model])),angdist,inverse=False)
		measures['{}-angdist'.format(model)] = results
		print(score(results[0],results[1]))
		
		results = run_experiment(SAME,DIFF,lambda x: max_vec(flair_embed_matrix(x,MODELS[model])),angdist,inverse=False)
		measures['{}-angmax'.format(model)] = results
		print(score(results[0],results[1]))

		#Point cloud
		print("Word cloud average method...")
		results=run_experiment(SAME,DIFF,lambda x: flair_embed_matrix(x,MODELS[model]),avg_sim_between_point_clouds)
		measures['{}-cloud'.format(model)] = results
		print(score(results[0],results[1]))

		#Point cloud kernel
		print("Word cloud kernel methods (third output is bandwidth)....")
		#gamma_range = np.arange(1,11,1)
		#bw_range    = 1.0/gamma_range[::-1]
		bw_range=[0.001,0.01,0.1,0.5,1.0]
		for bw in bw_range:
			try:
				results = run_experiment(SAME,DIFF,lambda x: flair_embed_matrix(x,MODELS[model]),partial(kernel_correlation_sim,bw=bw))
				measures['{}-bw-{}'.format(model,bw)] = results
				print(score(results[0],results[1]),bw)
			except Exception as e:
				print(bw, e)
			#print("bw={}".format(bw))
			#print(x+(bw,))

		
	with open("output_flair.csv", "w") as fh:	
		fh.write("\t".join(['metric','measure','which_comparisons']))
		fh.write("\n")
		
		for key in measures.keys():
			for xi in measures[key][0]:
				fh.write("{}\t{}\t{}\n".format(key,xi,'SAME'))
			for xi in measures[key][1]:
				fh.write("{}\t{}\t{}\n".format(key,xi,'DIFF'))

