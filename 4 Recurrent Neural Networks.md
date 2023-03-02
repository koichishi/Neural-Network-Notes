# Modeling Sequences - Recurrent Networks
## The issue: Time is important, and how do we represent time in connectionist networks?
- ## Approach 1: Map time into space (not RNN)
	- What is this?
		- Autoregressive models: Predict the next term in a seq from **a fixed number of previous terms** using "delay taps"
		- FFN: Generalize *autoregressive models* by using one or more layers of nonlinear hidden units
	- Works for simple problems
	- Doesn't work so well for arbitrary length items
	- However, transformers fuck them all over (in later lectures)
	- Summary:
		- Shared weights between positions: identical networks at every position
		- Longer sentences: add more prev terms
		- Term weights *interact* via attention mechanism
- ## Approach 2: Map time into the state of the network (**RNN!**)
	- What?
		- Use *activation memory* so that new things are processed based upon what has come before
		- This is called the network "state"
	- #### Variants:
		- Jordan networks: input $\rightarrow$ hidden $\rightleftarrows$ output
		- **Simple recurrent networks**, or Elman nets: input $\rightarrow$ hidden($t$) ($\rightleftarrows$ hidden($t-1$)) $\rightarrow$ output
			- This is called one-step backprop through time (BPTT(1)): changing the weights between the current and previous hidden states
			- An unrolled SRN:
			![[Pasted image 20230225141002.png]]
		- These networks can be used for many things, but SRNs typically are used for *prediction*
	- This is theoretically simple, and it is a *temporal* autoencoder ==what==
	- #### Types of #rnnproblems
		1. Prediction on the next word/pixel
		2. Seq generation: produce a word or sentence, caption an image
		3. Seq recognition: recognize a sentence, recognize an action
		4. Seq transformation: speech -> text, English -> French
		5. Learning a "program": sequential adder net, neural Turing machine
		6. Oscillations: walk talk, chew, fly
		...
	- #### Types of behavior
		- Oscillate is good for motor control
		- Settle to point attrctors: good for retrieving memories
		- Behave systematically: Good for learning transformations
		- (Able to) Behave chaotically: Bad for information processing?
		- RNNs could potentially learn to implement lost of small programs that each capture a nugget of knowledge and run in parallel, producing conflex effects
		...
	- ==(Question about the midterm)==
	- #### Back prop in time!![[Pasted image 20230216004105.png]]
		- Deltas are different per time, but weights are the same. We update by averaging over udpated weights
			- It is easy to modify the backprop to **incorporate tied weights**
			- Propagate the activations forward in time (call them $z_A(t), z_B(t), \dots$ )
			- Propagate deltas backwards
			- Now update (for example ) $W_3$, the weight from A to B as the avg of all of the weight changes: $$W_3\ +=\alpha\frac{1}{3}\sum^{3}_{t=1}\delta_B(t)z_A(t-1)$$
	- #### But RNN are hard to train at that time, LSTMs fixed this issue. Why?
		- In the forward pass we use nonlinear functions, which tends to limit the range of activations
		- The **backward pass is linear**. If double the error derivatives at the final layer, all the error derivatives will double! ==makes sense but isnt that the same as normal FNN?==
	- #### The problem of exploding or vanishing gradients
		- What happens to the magnitude when backprop?
			- Weights grows/shink exponentially through layers
			- In any case since these are the same weights, we are *iterating a linear system*
			- We can *avoid* this by init the weights carefully
		- Even with good init weights, RNNs have difficulty dealing with long-range dependencies
			- LSTMs make this!
	- #### In and out:
		- Inputs:
			- Initial states of all the units (e.g. the whole first layer of A, B, C)
			- Initial states of a subset of the units (e.g. just A at the first layer)
			- States of the same subset of the units at every time step - i.e. we have *inputs* at every time step (e.g. A's at every layer)
		- Teaching signals (target):
			- Final activities of all the units (e.g. the whole last layer of A, B, C)
			- Last few activities of all the units (e.g. the third last layer)
				- Good for learning attractors
				- Easy to add in extra error derivatives as we backprop
			- Activities of a subset of the units (e.g. B's at every layer)
				- Others are input or hidden units
	- Examples:
		- Language generation
		- Addition ==kinda vague... maybe too fast. is it important after all?==
	- ## [LSTM](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
		- The name of LSTM RNN refers to the analogy that a standard RNN has both "long-term" and "short-term" memories
		- #### Implementing a memory cell in a nerual network
			*To preseve info for a long time, we use a circuit that implements an analog memory cell: composed of a **keep** gate, a **write** gate and a **read** gate.*
			- A linear unit that has a self-link with a weight of 1 maintians its state 
			- Info stored in the cell by activating its write gate
			- Info retrieved by activating the read gate
		- #### Backprop gets easier!
			![[Pasted image 20230225134910.png]]
				$h_t^{l-1}:=\text{Input from the current time at lower layer (actual input)}$
				$h^l_{t-1}:=\text{Input from the previous time at current layer (from the last unit)}$
			**Note!!** LSTMs do not solve vanishing/exploding gradient problems, but provide an easier way for the model to learn long-distance dependencies.
	- #### Summary
		- To model sequential data, we can
			- Use input "buffering" to represent the sequence - mapping time into space, like NETTalk
			- Use *recurrence* in the network, mapping time into the state of the network
		- Recurrence can be implemented by unrolling the *network in time* to turn a recurrent net inoto a feedforward one
			- Early versions (Jordan and Elman) use simple architectures and unrolled one time step
			- We can envision the internal state space using PCA
		- Networks like this can be used in multiple ways: #rnnproblems
		- LSTM units allows the network to "latch" memory and hold it
		- [LSTM relieves the gradient problem of RNN](https://medium.datadriveninvestor.com/how-do-lstm-networks-solve-the-problem-of-vanishing-gradients-a6784971a577) in training
- ## Generative Modeling with RNNs
	- Motivation: We've had great tools for **fixed-size** data (vector in, vector out), but we want a system that output **structured data*
	- Idea: Compose complex actions as sequences of simple ones
		- Sentences might not admit a fixed-length representation, but words and characters can.
		- **Break up text** input/output to a word or character at each time step
	- Different usages?
		![[Pasted image 20230225160228.png]]
	- Generative text model: Shakespare generation (Sutskever et al., 2011)
	- Coorporate with other models: Image caption to text (Karpathy et al., 2014, Mao et al., 2014, Vinyals et al., 2014)
	- Supervised Character Model
	- #### Sequence to sequence (usage (d) e.g. translation)  
		- Seq2seq model ([Sutskever, et al. 2014](https://arxiv.org/abs/1409.3215)) used in Google translation in 2016. Normally, a Seq2seq model composed of
			- An **encoder** processes the input sequence and compresses the information into a context vector (aka sentence embedding or “thought” vector) of a *fixed length*. This representation is expected to be a good summary of the meaning of the *whole* source sequence.
			- A **decoder** is initialized with the context vector to emit the transformed output. The early work only used the last state of the encoder network as the decoder initial state.
			![[Pasted image 20230225211000.png]]
		- **Disadvantage** of this fixed-length context vector design is incapability of remembering long sentences. Often it has forgotten the first part once it completes processing the whole input. The *attention mechanism* was born ([Bahdanau et al., 2015](https://arxiv.org/pdf/1409.0473.pdf)) to resolve this problem.
