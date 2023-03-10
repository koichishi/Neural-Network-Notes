1. How do LSTMs deal with vanishing and exploding gradients? (2 points)

	**Answer:** Recall that in ordinary RNN, the gradient of the loss function w.r.t. the weight $\frac{\partial L}{\partial W}$ includes the exponential term of weight matrices, when back propagating through time. Therefore it grows/shrinks exponentially, causing the gradient quickly explodes/vanishes. 
	
	In LSTMs, cells get values by the additive write gate, pass values to the next layer by the read gate, and erase values by the forgot gate. The respective gradients control how much information is added/passed/forgotten. Additionally, the weight between connected LSTM cells is the identity matrix at the same layer. Therefore the exponential term of weight matrices $I^c=I,\forall c\in N$. Without the exponential growth/shrinkage it relieves the vanishing and exploding gradients that RNN has. 

2. This question refers to Figure 1. Give an example of a domain or problem where the following types of RNN architectures could be used: (2 points each)  
	- Figure 1b: many to one.  
	- Figure 1c: one to many  
	- Figure 1d: many in, then many out.  
	- Figure 1e: Simultaneous many in, many out.
	
	**Answer:**
	- Many to one: Sentiment classification - classifying if a given sentence has positive, negative or neutrual mood
	- One to many: Image captioning - given an image at one time step and generate description 
	- Many in then many out: machine translation
	- Simultaneous many in, many out: audio to text translation

3. What is the function of each of the gates in an LSTM cell? (2 points)
	
	**Answer:**
	- **Keep/forget** gate is a linear unit that has a self-link with identity weight matrix. It controls how much information is kept within the cell.
	- **Write** gate controls how much information is stored additively in the cell.
	- **Read** gate controls how much information that is stored in the cell is passed to the next layer.