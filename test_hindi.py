#Test Hindi 

import random
random.seed(1838)
import gensim

def load_hindi_data():
	same=[]
	diffa=[]
	diffb=[]
	with open("data/headline_pairs_hi.tsv","r") as fh:
		for line in fh:
			a,b=line.strip().split("\t",1)
			if random.random()<0.2:
				diffa.append(a)
				diffb.append(b)
			else:
				same.append([a,b])
	diff=[]
	for i in range(0,len(diffa)):
		a=random.randrange(0,len(diffa))
		b=random.randrange(0,len(diffa))
		if a==b:
			continue
		else:
			diff.append([diffa[a],diffb[b]])
			del diffa[a]
			del diffb[b]
	return same,diff

def clean(pairs, stopwords=None,subwords=False):
	count=0
	for i,pair in enumerate(pairs):
		for j in range(0,len(pair)):
			for key in list(pair[j].keys()):
				if (stopwords!=None and key in stopwords) or (subwords and key[0]=="#"):
					del pair[j][key]
					count+=1
			if len(pair[j].keys())==0:
				del pairs[i]
				break
	print("Removed {} words.".format(count))

if __name__ == "__main__":
	SAME,DIFF=load_hindi_data()
	
	from test_textsim import *
	from test_flair import *
	import pickle
	
	measures={}
		
	#XLM-R
	from flair.embeddings import XLMRobertaEmbeddings
	from flair.data import Sentence
	from wmdistance import wmdistance
	
	model="xlmr"
	xlmr=XLMRobertaEmbeddings()
		
	try:
		with open("hindi_same_dict.p","rb") as fh:
			SAME_xlmr_dict=pickle.load(fh)
	except:
		SAME_xlmr_dict=[[flair_embed_dict(e[0],xlmr),flair_embed_dict(e[1],xlmr)] for e in SAME]
		with open("hindi_same_dict.p","wb") as fh:
			pickle.dump(SAME_xlmr_dict,fh)
	
	try:
		with open("hindi_diff_dict.p","rb") as fh:
			DIFF_xlmr_dict=pickle.load(fh)
	except:
		DIFF_xlmr_dict=[[flair_embed_dict(e[0],xlmr),flair_embed_dict(e[1],xlmr)] for e in DIFF]
		with open("hindi_diff_dict.p","wb") as fh:
			pickle.dump(DIFF_xlmr_dict,fh)
	
	with open("data/stopwords-hi.txt", "r") as fh:
		stopwords=[line.strip() for line in fh]
	
	#clean(SAME_xlmr_dict,stopwords=stopwords,subwords=True)
	#clean(DIFF_xlmr_dict,stopwords=stopwords,subwords=True)

	SAME_words=[[list(pair[0].keys()),list(pair[1].keys())] for pair in SAME_xlmr_dict]
	DIFF_words=[[list(pair[0].keys()),list(pair[1].keys())] for pair in DIFF_xlmr_dict]
	
	#Set distance
	print("Set intersection over set union...")
	results = run_experiment(SAME_words,DIFF_words,lambda x:x, set_dist)
	measures['set'] = results
	print(score(results[0],results[1]))
			
	#Word mover's distance
	results = run_experiment(SAME_xlmr_dict,DIFF_xlmr_dict,lambda x: x,	wmdistance,inverse=True)
	measures['{}-wmdist'.format(model)] = results
	print(score(results[0],results[1]))
	

	SAME_xlmr_matrix=[[flair_embed_matrix(e[0],xlmr),flair_embed_dict(e[1],xlmr)] for e in SAME]
	DIFF_xlmr_matrix=[[flair_embed_matrix(e[0],xlmr),flair_embed_dict(e[1],xlmr)] for e in DIFF]


	SAME_xlmr_matrix=[[np.array(list(e[0].values())),np.array(list(e[1].values()))] for e in SAME_xlmr_dict]
	SAME_xlmr_dict=None
	DIFF_xlmr_matrix=[[np.array(list(e[0].values())),np.array(list(e[1].values()))] for e in DIFF_xlmr_dict]
	DIFF_xlmr_dict=None

	results = run_experiment(SAME_xlmr_matrix,DIFF_xlmr_matrix,lambda x: max_vec(x),distance.cosine,inverse=True)
	measures['{}-cosmax'.format(model)] = results
	print(score(results[0],results[1]))

	results = run_experiment(SAME_xlmr_matrix,DIFF_xlmr_matrix,lambda x: mean_vec(x),distance.cosine,inverse=True)
	measures['{}-cosine'.format(model)] = results
	print(score(results[0],results[1]))

	results = run_experiment(SAME_xlmr_matrix,DIFF_xlmr_matrix,lambda x: mean_vec(x),angdist,inverse=False)
	measures['{}-angdist'.format(model)] = results
	print(score(results[0],results[1]))
	
	results = run_experiment(SAME_xlmr_matrix,DIFF_xlmr_matrix,lambda x: max_vec(x),angdist,inverse=False)
	measures['{}-angmax'.format(model)] = results
	print(score(results[0],results[1]))

	#Point cloud
	print("Word cloud average method...")
	results=run_experiment(SAME_xlmr_matrix,DIFF_xlmr_matrix,lambda x: x,avg_sim_between_point_clouds)
	measures['{}-cloud'.format(model)] = results
	print(score(results[0],results[1]))

	#Point cloud kernel
	print("Word cloud kernel methods (third output is bandwidth)....")
	#gamma_range = np.arange(1,11,1)
	#bw_range    = 1.0/gamma_range[::-1]
	bw_range=[0.001,0.01,0.1,0.5,1.0]
	for bw in bw_range:
		try:
			results = run_experiment(SAME_xlmr_matrix,DIFF_xlmr_matrix,lambda x: x,partial(kernel_correlation_sim,bw=bw))
			measures['{}-bw-{}'.format(model,bw)] = results
			print(score(results[0],results[1]),bw)
		except Exception as e:
			print(bw, e)
		#print("bw={}".format(bw))
		#print(x+(bw,))

		
	with open("output_hindi.csv", "w") as fh:	
		fh.write("\t".join(['metric','measure','which_comparisons']))
		fh.write("\n")
		
		for key in measures.keys():
			for xi in measures[key][0]:
				fh.write("{}\t{}\t{}\n".format(key,xi,'SAME'))
			for xi in measures[key][1]:
				fh.write("{}\t{}\t{}\n".format(key,xi,'DIFF'))		
