1. How do LSTMs deal with vanishing and exploding gradients? (2 points)
	**Answer:** Recall that in ordinary RNN, the gradient of the loss function w.r.t. the weight $\frac{\partial L}{\partial W}$ includes the exponential term of weight matrices, when back propagating through time. Therefore it grows/shrinks exponentially, causing the gradient quickly explodes/vanishes. 
	In LSTMs, cells have additive write coming from the write gate. The weight between connected memorized cells is the identity matrix. Therefore the exponential term of weight matrices $I^c=I,\forall c\in N$. Without the exponential growth/shrinkage it relieves the vanishing and exploding gradients that RNN has. 

2. This question refers to Figure 1. Give an example of a domain or problem where the following types of RNN architectures could be used: (2 points each)  
	- Figure 1b: many to one.  
	- Figure 1c: one to many  
	- Figure 1d: many in, then many out.  
	- Figure 1e: Simultaneous many in, many out.
	**Answer:**
	- Many to one: Sentiment analysis - classifying if a sentence has positive, negative or neutrual mood
	- One to many: Image captioning - given an image at one time step and generate description 
	- Many in then many out: Machine translation
	- Simultaneous many in, many out: Audio to text translation

3. What is the function of each of the gates in an LSTM cell? (2 points)
	**Answer:**
			- **keep/forget** gate: a linear unit that has a self-link with identity weight maintians its state
			- **write** gate: values are stored additively in the cell by activating its write gate
			- **read** gate: values are outputed by activating the read gate