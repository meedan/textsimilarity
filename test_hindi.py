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

if __name__ == "__main__":
	SAME,DIFF=load_hindi_data()
	
	from test_textsim import *
	from test_flair import *
	
	measures={}
		
	#XLM-R
	from flair.embeddings import XLMRobertaEmbeddings
	from flair.data import Sentence
	from wmdistance import wmdistance
	
	MODELS={"xlmr":XLMRobertaEmbeddings()}
	model="xlmr"
	
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

		
	with open("output_hindi.csv", "w") as fh:	
		fh.write("\t".join(['metric','measure','which_comparisons']))
		fh.write("\n")
		
		for key in measures.keys():
			for xi in measures[key][0]:
				fh.write("{}\t{}\t{}\n".format(key,xi,'SAME'))
			for xi in measures[key][1]:
				fh.write("{}\t{}\t{}\n".format(key,xi,'DIFF'))		
