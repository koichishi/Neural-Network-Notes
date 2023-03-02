- ## How to deal with overfitting
	- ### The best way: *Get more data!*
	- ### Minimize model 
		- $J=E+\lambda C$ where $E$ is the error and $C$ is the model complexity (*regularization*)
		 General Idea: Ockham's Razor: Make model "smaller"
		- $L_2$ regularization: Minimize $||W||^2_2$
			Derivative: $2w$ - Penalizes bigger weights more  
		- $L_1$ regularization: Minimize $|W|$
			Derivative: 1 - Penalizes weights smaller at a constant rate
		- Rumelhart's idea: Minimize $C=||W||^2/(||W||^2+1)$
			Penalizes big weights less while small weights more
	- ### Dropout: Randomly turn off hidden units during training
		- For any hidden layer, probabilitiscally *turn off* some fraction of the hidden units
		- It makes hidden unit more independent
		- Explores an exponential number of models
	- ### Early stopping
		- Have a hold out set (fraction of the training set) - a stand-in for the unseen test set.
		- Use the remaining portion to change the weights. 
		- Watch the error on the holdout set and stop when it starts to rise
	- ### Add small gaussian random noises to the inputs/hidden units 
		- Improve generalization!

- ## Stochestic Gradient Descent vs Batch Gradient Descent
	- ### BGD
		- ### Steps:
			1. Get one example
			2. Compute the gradient
			3. Add it to a running avg of the gradients
			4. If all been seen, change the weights
			5. Go to 1
		- This follows the true gradient for the error function
		- BUT, if the data is redundant, model learns a lot at beginning but less later -> waste of computation
	- ### SGD - an online learning method
		- ### Steps:
			1. Get one example
			2. Compute the gradient
			3. Change the weights
			4. If all been seen, reshuffle them
			5. Go to 1
		- Stochastic due to *re-shuffle*. The actual gradients you follow are in random order
		- Not following the true gradient - sometimes get around local mim
	- ### Why using either?
		- Why SGD?
			- Tends to learning faster due to redundancy in the training set - you can learn a lot about the problem well before seeing all example
			- Tends to get better solutions & generalization
			- Adapt to a changing environment
			- Can be used for VERY large datasets due to its online nature
		-  Why Batch learning?
			- Convergence conditions are better understood - in general, there's a fair amount of theory
			- Better optimization methods exists - *although* since 2012 all sorts of new online approaches have been developed
			- *However*, this also leads to large weights -> poor generalization
		- Why Mini-Batch learning?
			- Compute batches in parallel with the same weights and update the weights once
			- Combining the two's avdantage
			- Efficient on GPUs
	- ### Shuffle the example (SGD)
		- Shuffling means that we will see parttern from diff classes
		- It is important that a minibatch have most of the categories in it, so the model learns to discriminate the classes
		- A possible heuristic: present examples with more error more often. These examples are hard due to:
			1. miss-labeled data
			2. learn more from ez examples before training on the hard ones
			3. but, this approach can improve performance on infrequent 
		- In SGD, should we shuffle minibatch before computing the gradients?
			- NO! Shuffle should happen before partitioning into minibatches, or it will have no impact on the weight changes!
- ## PCA
	- Pop-up quesiton: 
		- What's wrong if all positive inputs?
			- $\delta=E'(w^Tx)$ has the same sign for every weights -> all weights change in the same direction!
		- What's wrong with correlated inputs?
			- They encode similar information
			- If we do PCA, some variables will represent the correlated features
		- What's wrong with very diff scale of inputs?
			- Bias towards big-scale examples
	- #### Why PCA?
		- PCA shifts the mean of the input -> not all +/- examples!
		- Decorrelates the input
		- Safe only the most significant dimensions
		- (Not part of PCA but good for NN) Divide by the standard deviation, makes them roughly the *same size*
	- #### Diff from z-scoring
		- Does not decorrelate the input - because z-scoring applied to each variable independently
		- Cannot throw dimensions - because dont compute eigenvectors
		- Divides by the standard deviation, make all variables mean 0 & unit standard deviation.
	- #### Side Node #1: A linear autoencoder essentially does PCA
		- It's minimizing the same thing: squared error
		- A hidden unit is like a principal component, and so the input is projecting onto that is its coordinate on that component
		- But the variance is spread across all of them
- ## Change the Sigmoid...?
	- It changes outputs to positive again!
	- Recommended a 0-oriented sigmoid
		- BUT ReLU seems work fine (mystery)
- ## Weight Initialization
	- Can't be all zero. Why?
		Recall $w_{ij}+=\alpha\delta_jx_i$, so outgoing weights from each input will be the same, and the hidden units will all compute the *same* feature!
	- ==We want the weighted sum $a_j$ of the inputs to be in the linear range of the **sigmoid**:==
		1. gradients will be largest and
		2. the network can learn any linear part of the mapping before the non-linear part
		- So we want $a_j$ to be 0 mean and std deviation (with the recommended sigmoid)
			- Assuming right sigmoid, the input layer weight has 0 mean and unit std, to **ensure the output of input layer to have 0 mean and unit std**, the next weights should be init as mean=0 and std=$1/\sqrt m$ where $m$ is the *fan-in*, the number of input to the current node  
			- This also make sure the output of the first layer to be 0! (See slides)
		- This whole thing means: 
	- For ReLU, $W~N(0,\sqrt{\frac{2}{n_l}})$ where $l$ is the fan-in
	- But all this careful work won't survive the weight changes during the learning! We need **Batch Normalization**
- ## Batch Normalization
	- Uses all of the weight init insights in a *dynamic sense*!
	- Takes a minibatch and
		1. z-scores each var (each input $a_j$) at every layer of the network individually over the mini-batch (This is actually a weird NN layer that does normalization) $$
		\hat{a_i}=\frac{a_i-\mu_i}{\sqrt{\sigma_i^2+\epsilon}}$$ 
			Where $\mu_i$ and $\sigma_i^2$ are the mean and variance of $a_i$ over the mini-batch
		2. Gives the network the chance to *undo* batch normalization by giving it adaptive parameters $$
		\tilde a_i=\gamma_i\hat a_i+\beta_i$$
			Where $\gamma_i$ and $\beta_i$ are learnable parameters. So, the original $a_i$ is replaced by $\tilde a_i$ . This can be back-propagated through as its differentiable
- ## Learning methods for neural networks
	- ### Momentum
		- Instead of using gradient to change the weights, keep a *running average* of the weight changes
		- Use separate adaptive learning rates for *each* parameter
			- Slowly adjust the rate using the consistency of the gradient for that parameter
		- Reminder:
			- For a linear neuron with MSE, the error surface is concave
			- For multi-layer non-linear nets the error surface is *much* more complicated - with local minima, long plateaus, saddle points, etc.
		- Convergence speed of full batch learning when the error surface is a quadratic bowl: Going steepest descent does not point at the minimum unless the ellipse is a circle 
		- We want to move in direction also on behave of the previous gradients:
			- The effect of the gradient is to increment the previous average. The average also decays by $\alpha$ which is slightly less than 1
		 $$\text{v}(t)=\alpha\text{v}(t-1)-\epsilon\frac{\partial E}{\partial \text{w}}(t)$$
			 - The weight change is equal to the current average
			$$
			\begin{align}
			\Delta\text{w}(t) &= \text{v}(t) \\
			&= \alpha\text{v}(t-1)-\epsilon\frac{\partial E}{\partial\text{w}}(t) \\
			&= \alpha\Delta\text{w}(t-1)-\epsilon\frac{\partial E}{\partial\text{w}}(t)
			\end{align}
			$$
			The weight change can be expressed in terms of the previous weight change and the current gradient
		- Behavior of the momentum method:
			- At the beginning of learning, the gradient might be large
				- Start with a small momentum (e.g. 0.5)
				- Once the large gradient have disapperaed and the weights are stuch in a ravine the momentum will increase smoothly
		- A better type of momentum
			- Standard momentum
				- First computes the current gradient, 
				- Then takes a big jump in the direction of the updated accumulated gradient
			- Ilya Sutskever Method (intuition: correct a mistake after you have made it!)
				- First make a big jump in the direction of the previous accumulated gradient; 
				- Then measure the gradient where you end up and make a correction
	- ### Adaptive learning rates
		- In a multilayer net, the appropriate learning rates can vary widely between weights
			- The magnitudes of the gradients are often very different for different layers
			- The fan-in of a unit determines the size of the "overshoot" effects: changing many of  the weights of a unit to correct the *same* error
		- So use a global learning rate (set by hand) multiplied by an appropriate local gain for each weight
		- One way to determine the individual learning rates
			$$
			\Delta w_{ij}=-\epsilon\ g_{ij}\ \frac{\partial E}{\partial w_{ij}}
			$$
			- Start with a local gain of 1 for every weight ($g_{ij}=1$)
			- Increase the local gain if the gradient for that weight doesn't change sign, vise versa
				- Small additive increases ($+0.05$) and multiplicative decreases ($*0.95$)
				- Limit the gains in a range
			- Use full batch learning or big mini-batches to ensure sign not alter due to sampling error of mini-batch 
		- Adaptive learning rates only deal with *axis alignment* effect while momentum does not - we want an adaptive learning method to combine  
	- ### rprop: Using only the sign of the gradient
		- For full batch learning,
			- Increase the step size for a weight *multiplicatively* if last two signs agree; decrease *multiplicaively* otherwise
			- Limit the step size to be less than 50 and more than a millionth
		- Why not work with mini-batch? 
			- SGD wants that when the learning rate is small, it averages the gradients over successive mini-batches
			- But if 9 batches get gradient of $+0.1$ and 1 batch gets $-0.9$, rprop will increase the weight $+0.8$ as it cares only the sign
		- Lets combine mini-batch & rprop!
	- ### rmsprop
		- The problem with mini-batch rprop is that we divide by a different number for each mini-batch (we divide by the size of *gradient* instead of the batch)
		- Keep a moving average of the squared gradient for each weight$$
		 MeanSquare(w,t)=0.9*MeanSquare(w,t-1)+0.1*(\frac{\partial E}{\partial w}(t))
		 $$
			 Dividing the gradient by $\sqrt{MeanSquare(w,t)}$
	 - ### Summary of learning methods:
		 - For small datasets or bigger without much redundancy, use full-batch
			 - Conjugate gradient, LBFGS ...
			 - adaptive learning rates, rprop ...
		 - For big, redundant dataset, use mini-batches
			 - Try gradient descent with momentum
			 - Try the adam optimizer
			 - Try whatever Yann LeCun is doing 
		 - Why no simple recipe: 
			 - Neural nets differ a lot
			 - Tasks differ a lot 
