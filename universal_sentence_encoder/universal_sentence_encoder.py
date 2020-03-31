import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import scipy.spatial.distance
import math

class UniversalSentenceEncoder(object):
	def __init__(self,model_path="https://tfhub.dev/google/universal-sentence-encoder-large/5"):
		self.model_path=model_path
		self.model=hub.load(model_path)

	def vectorize(self,doc):
		"""
			vectorize: Embedd a text snippet in the vector space.
			If doc is a list, the return value will be a matrix with each row corresponding to one element in the list.
			If doc is not a list, then the return value will be a vector.
		"""
		if isinstance(doc,list):
			return self.model(doc).numpy()
		else:
			return self.model([doc]).numpy()[0]
    
	def angular_simularity(self,u,v):
		"""
		Return the angular simularity between two vectors u and v.
		@return integer between zero and one inclusive. One indicates identicial
		"""
		#Ensure cosine is between zero and one
		cosdist=max(0,min(scipy.spatial.distance.cosine(u,v),1))
		return 1-math.acos(1-cosdist)/math.pi


if __name__=="__main__":

	finalsent="What is your age"
	messages = [
		"I like my phone",
		"Your cellphone looks great",
		"Will it snow tomorrow",
		"Hurricanes have hit the US",
		"How old are you",
		finalsent,
	]
	
	#This is the mutlilingual model
	#16 languages (Arabic, Chinese-simplified, Chinese-traditional, English, French, German, Italian, Japanese, Korean, Dutch, Polish, Portuguese, Spanish, Thai, Turkish, Russian)
	#import tensorflow_text
	#encoder=UniversalSentenceEncoder(model_path="https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")
	
	#This is the English-only model
	encoder=UniversalSentenceEncoder()
	
	embeddings=encoder.vectorize(messages) #Check list
	sent_embedding=encoder.vectorize(finalsent) #Check string
	
	dists=np.zeros((len(messages),len(messages)))
	for i in range(0,len(messages)):
		for j in range(0,len(messages)):
			dists[i,j]=encoder.angular_simularity(embeddings[i],embeddings[j])

	print(dists)
	
	#The upper triangle and bottom triangle ought to be the same
	#because distances are symetric
	assert np.allclose(dists, dists.T), "Distance matrix is not symmetric"

	np.fill_diagonal(dists,0) #Check the below are true ignoring distance to self.
	assert dists[0,:].argmax()==1, "The closest sentence to index 0 is not index 1"
	assert dists[2,:].argmax()==3, "The closest sentence to index 2 is not index 3"
	assert dists[4,:].argmax()==5, "The closest sentence to index 4 is not index 5"
	
	assert encoder.angular_simularity(embeddings[5],sent_embedding)==1.0,"Different vectors for same sentence"
	
	print("All good")
	
