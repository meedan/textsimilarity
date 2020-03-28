
#pip install tensorflow tensorflow-hub tensorflow_text>=2.0.0rc0 pyyaml h5py

def angdist(u,v):
	return 1-acos(1-distance.cosine(u,v))/pi

if __name__ == "__main__":
	import numpy as np
	import tensorflow as tf
	import tensorflow_hub as hub
	import tensorflow_text
	
	#for distance calc
	from math import acos, pi
	from scipy.spatial import distance


	unisent_multilingual = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")
	
	sentence1 = "Obama speaks to the media in Illinois"
	sentence2 = "The president greets the press in Chicago"
	
	embedding1 = unisent_multilingual([sentence1]).numpy()
	embedding2 = unisent_multilingual([sentence2]).numpy()
	
	print(angdist(embedding1,embedding2)) #0.7189
	
	sentence3 = "We will win and defeat coronavirus"
	embedding3 = unisent_multilingual([sentence3]).numpy()
	
	print(angdist(embedding1,embedding3)) #0.5289
	

