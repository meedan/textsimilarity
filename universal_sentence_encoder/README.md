# Universal Sentence Encoder

This is a minimal working example of the Universal Sentence Encoder.

Further detail on the model is available on 
[TensorFlow Hub](https://tfhub.dev/google/universal-sentence-encoder-large/5)
and in the original academic paper: 
[Universal Sentence Encoder](https://arxiv.org/pdf/1803.11175.pdf).

`universal_sentence_encoder.py` implements one class, `UniversalSentenceEncoder`.
The class expects the location (e.g., URL or local filesystem) of a 
TensorFlow Hub model. It defaults to the URL above.

The class provides two methods:
* `vectorize` accepts either a string or a list of strings. 
 If the input is a string, it returns an array with the vector embedding of the string.
 If the input is a list, it returns a matrix with  each row corresponding to one element in the list.

* `angular_simularity` compares how similar two vectors are.
 It is an implementation of Equation 1 in the paper linked above.

![sim(u,v)=1-\text{arccos}(\frac{u \cdot v}{\left\lVert u\right\rVert \left\lVert v\right\rVert})/\pi](https://render.githubusercontent.com/render/math?math=sim(u%2Cv)%3D1-%5Ctext%7Barccos%7D(%5Cfrac%7Bu%20%5Ccdot%20v%7D%7B%5Cleft%5ClVert%20u%5Cright%5CrVert%20%5Cleft%5ClVert%20v%5Cright%5CrVert%7D)%2F%5Cpi)
 

