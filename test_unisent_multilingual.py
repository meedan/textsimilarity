
#!pip uninstall --quiet --yes tensorflow
#!pip install --quiet tensorflow-gpu
#!pip install --quiet tensorflow-hub
#!pip install tensorflow_text>=2.0.0rc0
#!pip install -q pyyaml h5py


#pip install tensorflow tensorflow-hub tensorflow_text>=2.0.0rc0 pyyaml h5py


if __name__ == "__main__":
	import numpy as np
	import tensorflow as tf
	import tensorflow_hub as hub
	import tensorflow_text


	unisent_multilingual = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")

	#embedding1 = description_embeddings(["The quick brown fox jumps over the lazy dog"])
	#print(embedding1.numpy())

	from test_textsim import *
	
	measures={}
	
	SAME,DIFF=load_data("./data/test_STS2017en-en.txt")
	print("Multilingual universal sentence encodings")
	
	results = run_experiment(SAME,DIFF,lambda x : unisent_multilingual([x]).numpy(),angdist,inverse=False)
	measures['multi-unisent-angdist'] = results
	print(score(results[0],results[1]))

	results = run_experiment(SAME,DIFF,lambda x : unisent_multilingual([x]).numpy(),distance.cosine,inverse=True)
	measures['multi-unisent-cosine'] = results
	print(score(results[0],results[1]))
	
	results = run_experiment(SAME,DIFF,lambda x : unisent_multilingual([x]).numpy(),sqrtdist,inverse=True)
	measures['multi-unisent-sqrt'] = results
	print(score(results[0],results[1]))
		
	with open("output_multilingual_unisent.csv", "w") as fh:	
		fh.write("\t".join(['metric','measure','which_comparisons']))
		fh.write("\n")
		
		for key in measures.keys():
			for xi in measures[key][0]:
				fh.write("{}\t{}\t{}\n".format(key,xi,'SAME'))
			for xi in measures[key][1]:
				fh.write("{}\t{}\t{}\n".format(key,xi,'DIFF'))
		
		#df = pd.DataFrame(tuples, columns=['metric','measure','which_comparisons'])  
		#df.to_csv(outfile,index=False)

