All the derivation can be found in the slides under [Piazza\\Resources](https://piazza.com/ucsd/winter2023/cse251b_wi23_a00/resources)

# Supervised Learning
## Linear Regression
- Set of inputs and associated outputs
- Assume there exists a linear function been corrupted by Gaussian Noises
- **Find the correct weights**
	- Minimize the Sum Squared Error (SSE):
		$$SSE = \frac{1}{2} \sum^N_{n=1}(\sum^d_{j=0}w_j\ x_j^n-t^n)^2$$
		$N$ is the number of examples or observations or *patterns*, $x^n$ is the $n^{th}$ example ($x^n \in R^d$) and $t^n$ is the target for the $n^{th}$ example.
	- This is the square of the *Euclidean distance*
	- To solve it:
		- take derivative w.r.t. $\vec{w}$ and set the eq = 0
		- $\vec{w}=X^\dagger\vec{t}$, where $X^\dagger=(X^TX)^{-1}X^T$ is the *pseudoinverse* of *X*
		- $X^\dagger$ hard to compute -> Gradient Descent!
- ### [Gradient Descent](https://blog.skz.dev/gradient-descent) for Linear Regression
	- Recall the model is $y(x)=g(x)$, where $g$ is the activation function, For linear regression, $g$ is the identity function: $g(a)=a$, where $a^n=w^T\ x^n=\sum^d_{i=0}w_i\ x_i^n$ is the weighted sum of the inputs for pattern *n*
	- #### Steps:
		1. Inspect the steepest slope at the current position
		2. Walk $\alpha\times\bar g$ in that direction, where $\bar g$ is the gradient of the **loss function**
		3. If still descending, repeat step 1, otherwise stop!
	- For Linear Reg: we choose loss function to be the distance btwn the targets and the outputs - Mean Squared Error(MSE) $MSE=\frac{1}{N}\sum_{n=1}^N(t^n-y^n)^2$ 
		- Move params in the direction of the negative slope:
			$w_i=w_i-\alpha\ \frac{\partial{MSE}}{\partial{w_i}}$
		- Solve for the eq we get the **delta rule** (gradient descent learning rule for updating the weights of the inputs in a single layer NN)
			$w_i=w_i+\alpha\frac{1}{N}\sum_{n=1}^N\partial^nx_i^n$, let $\partial^n=(t^n-y^n)$ 
		- Intuitively makes sense: e.g. If $t^n$ is bigger than $y^n$, we want larger $y^n$ => if $x_i^n$ is positive, this increases the weight
	- ### Two variants:
		1. Batched Gradient Descent: Update once a batch of patterns
		2. Stochastic Gradient Descent: Randomize the order of patterns
			- Faster than BGD cuz lots of data sets can be redundant

## Perceptron
Compute a decision boundary $y(\text{x})=0$. A $d-1$ dimensional hyperplane in a $d$ dimensional input space => can only discriminate *linearly separable categories*
- For each layer:
$$
f(x)=
\begin{cases}
1 & \quad \text{if $y(\text{x})=\sum_{i=0}^dw_ix_i \geq 0$}\\ 
0 & \quad \text{otherwise}
\end{cases}
\quad
x_0\overset{\Delta}{=}1
$$
			 (FYI we cannot do XOR in single layer perceptron)center
- Learning rule assuming $\alpha=1$:
	$w_i=w_i+\delta x_i$, where $\delta=(t-y)$
- ### Characteristics:
	- Error correction learning: learning only on errors
	- Slow!
	- **w** is orthogonal to **x** =...=> $w_1...n$ specify the orientation of the decision boundary and $w_0$ specifies its location along the weight vector (proof in PA1)
	- No activation function for $y(x)$!
- ###  Multiple Categories - say $C$ categories
	- Prep $C$ weights pick the *max* (can use different methods like softmax) $y_c$
	- Name $R_i$ to be the region of $y$ belongs to $C_i$
	- Properties:
		- Every region is *convex*: If $x_A$ and $x_B$ are in one region, then so is any point btwn them
- ### Problems with perceptrons
	- Its learning rule guarantee: anything a perceptron can *compute*, it can then *learn to compute*
	- Problem, lots of things were not computable (e.g. XOR)
	- What about hidden units?
		- If you had hidden units, you could compute *any* boolean function 
		- But no learning rule exists so far (at that point) that has that perceptron guarantee
- ### Aside about perceptrons
	- No hidden units - but assume nonlinear preprocessing
	- Hidden units compute features of the input <-> Nonlinear preprocessing is a way to choose feature by hand
	- Support Vector Machines(SVM) do this in a principled way, followed by a highly sophesticated perceptron learning algorithm

## Logistic Regression
A generalization of the linear classifier and the perceptron
Bernoulli dist. instead of 0/1 as perceptron does!
- A monotonic activation function $g()$:
	$y(x)=g(w^Tx+w_0)$
	where $g(w)=\frac{1}{1+e^{-x}}$ the logistic function
- This is still considered a linear classifier, because since $g$ is monotonic, the boundary will still be linear
- We can treat the output as **posterior probabilities** $P(c_1|x)=y=g(a)=\frac{1}{1+e^{-x}}$ 
- That's nice, BUT:
	- Can't assume data follows Gaussian
	- Need to learn the *weights*
	- There is no closed form for the weights
	=> Gradiant Descent!
- ### [Gradient Descent](https://blog.skz.dev/gradient-descent) for Logi. Reg.
	- (Reminder of the notation)
		- $y=g(a)$ is the output of the network
		- $g(a)$ is the *activation function*
		- $a$ is the weighted sum of the inputs (the *net input*) $a=\sum^d_{j=0}w_jx_j$ 
	- First intuition: MSE => bad learning rate!
			$w_i=w_i-\alpha\ \frac{\partial{MSE}}{\partial{w_i}}$ 
		 => $w_i=w_i+\frac{\alpha}{N}\sum_{n=1}^N(t^n-y^n)g'(a^n)x_i^n$
			 -- Cool thing about logit $g'(a)=g(a)(1-g(a))$ --
		 => $w_i=w_i+\frac{\alpha}{N}\sum_{n=1}^N(t^n-y^n)g(a)(1-g(a))(a^n)x_i^n$
			**Why slow? Notice that when $g(a)$ is close to 0/1 (i.e. confident) but wrong, the change step would be tiny** 
	- Finally, cross-entropy!

## Softmax - Generalization of logistic regression to multiple categories
Compute the weighted sum of the inputs to each category output, $a_k$
(Can compute all at once efficiently: $\vec a=W^T\vec x$ where $W$ is the *weight matrix*)
- The softmax activation function - also known as the *softmax distribution*: 
	$$
\begin{align}
y_k 
&=g(a_k) =\frac{e^{a_k}}{\sum_{j=1}^ce^{a_j}}>0 \\
\sum_{k=1}^cy_k &= 1
\end{align}
	$$
	- Notice :
		- $y_k>0$ no matter what $a_k$ is
	- **Mutually exclusive categories** - not always a good thing - might want to use **logistic outputs** for independent outputs (e.g. tagging something to be in more than one categories) 
- We can treat the output as **posterior probabilities** $P(c_k|x)=y_k=g(a_k)=\frac{e^{a_k}}{\sum_{j=1}^ce^{a_j}}$ 
### More activation funcitons
- ReLU, Leaky ReLU
- ELU
- Tanh
- ...

## Forward Propagation
Consists of applying activation funciton at each layer
hiddens could be ReLU - the outputs softmax
- Three layer model (input -> hidden -> output) are universal approximator
- Why deeper ones? less hidden units!

## Maximum Likelihood ==diff between bnn and ffn?==
- #### given the data $D$, which params $W$ of our model are most likely? 
$$\arg\max_W P(W|D)=\arg\max_W \frac{P(D|W)P(W)}{P(D)}=\frac{\text{Likelihood}\times\text{Prior}}{\text{normalizing constant}}$$
	where the prior initially should be flat, and by Bayes' theorem, instead, we should figure out how to maximize the likelihood of the data given the weights, $$\arg\max_WP(W|D)=\arg\max_WP(D|W)=\arg\max_M \prod_{n=1}^Np(x^n)$$
- #### In neural net, ==midterm question!==
	$$L=\prod^N_{n=1}p(x^n,t^n)=\prod^N_{n=1}p(t^n|x^n)p(x^n)$$
	taking negative log,$$-\ln L=\sum^N_{n=1}(p(t^n|x^n)+\ln p(x^n))$$
	the last term $\ln p(x^n)$ is constant and do not effect the minimization.
	**We assume**:
	1. targets given observations $p(t^n|x^n)$ follows gaussian distribution 
	2. noise $\epsilon$ follows noise with 0 mean
	3. there exists a true deterministic function $h(x^n)$
	=> $t^n=h(x^n)+\epsilon$ 
	Thus by plugging in the gaussion eq and simplification, we only need to minimize $$\frac{1}{2}\sum^N_{n=1}(t^n-y(x^n;w))^2$$
	=> The SSE!
- #### Another example on other distribution maybe?
	- We want the network to produce the prob. that the input is in cat 1 $y(x^n)=P(C_1|x^n)$ 
	- It follows Bernoulli distribution
	- Let $t^n=0$ if in cat 1 and 1 if in cat 2. The prob. of one point can be written as$$p(t^n|x^n)=(y^n)^{t^n}(1-y^n)^{1-t^n}$$
		same procedure as negative log likelihood,
		reaches a minimum when $t^n=y^n$ (as expected)

## Cross-entropy
- ### What is *entropy* anyway?
	- A measure of info in a msg about a random variable: $-P(x)\log_2 P(x)$
	- OR we can say that entropy measures the unpredictability or impurity of the system
	- Telling the value of a denser (less variance) distribution gives less information than one from a higher variance
- ### The entropy of a distribution is $-\sum_{n=1}^N P(x^n)\ln P(x^n)$ 
	- E.g. Let's understand it with an example where we have a dataset having three colors of fruits as red, green, and yellow. Suppose we have 2 red, 2 green, and 4 yellow observations throughout the dataset. Then:
		$P_r$ = 2/8 =1/4 
		$P_g$ = 2/8 =1/4
		$P_y$ = 4/8 = 1/2
	 => Entropy = $-(-0.5-0.5-0.5)=1.5$  
- ### Now, *cross-entropy* is how much I lose by sending the msg from one distribution using another distribution: $XENT=-\sum_{n=1}^NP(t^n)\ln P(y^n)$ 
	- It is best if the two distributions are the same - so we minimize this by controlling $P(y^n)$  
	- We can think of X-ent as trying to move the network's output distribution closer to the target distribution
- ### Use in Multinomial Regression
	- We want $P(C_k|x^n)=y_k(x^n)$ and $t^n_k=1$ if the $n^{th}$ example is from cat $k$. How to write its likelihood?		$$P(t^n|x^n)=\prod_{k=1}^c(y_k^n)^{t_k^k}$$
		and the negative log likelihood is 		$$-\ln L=\sum_{n=1}^N\sum_{k=1}^c-t_k^k\ \ln y_k^n$$
		WOW! The cross-entropy!
- To recap:
	When doing c-way classification AKA multinomial regression, if we
	1. assume the targets are multinomially distributed, and 
	2. maximize the likelihood of the data by minimizing the negative log likelihood
	We find we need to minimize the Cross Entropy Error!
### Another approach to objective functions
- Sometimes we aren't trying to fit a distribution. Instead we may want to cluster the data in a lower dimensional space
	- i.e. take the input and map it into a space where it is close to others from the same cat
	- This *could* be supervised in some models (e.g. Siemeses networks) if we know the cat; and unsupervised otherwise
### Summary for Maximum Likelihood
- A Gaussian distribution leads to minimize SSE
- A Bernoulli distribution leads to minimize cross-entropy
- A multinomial distribution, with the right target coding, leads to cross entropy
- Other kinds of objective functions are possible, too

## Back Propagation
- (Enter Rumelhart, Hinton, & Williams 1985) Works a lot like the perceptron algorithm:
	- Randomly choose an input-output pattern
	- Present the input, let activation propagate thru the network
	- Give the *teaching signal*
	- propagate the error back thru the network and change the connection strengths
- Uses chain rule of calculus to go downhill in an error measure with respect to the weights
- FYI relatively robust to programming error 
### Back to Forward Propagation
Hidden unit: compute the nonlinear activation function (e.g. logistic, tanh, ReLU) $z_j=g_h(a_j)$
Output unit: 
	1. compute the weighted sum of the inputs (net input) $a_k=\sum^M_{j=0}w_jz_j$ and 
	2. compute the output value with actination function $y_i=g_{out}(a_k)$ 
### How to do Back Propagation - Gradient Descent!
**Notation:** 
	- $z_i$ mean either the output of a hidden unit $z_i=g(a_i)$ or an input $z_i=x_i$
	- $a_j=\sum_iz_i*w_{w_{ij}}$ is the weighted sum from $i$'s to unit $j$
	- $J$ mean the objective function
- Derivation can be found in the slides
	We **define** $\delta_j=-\frac{\partial J}{\partial a_{j}}$ and **found** $\frac{\partial a_j}{\partial w_{ij}}=z_j$
	So we have the delta rule! $$w_{ij}=w_{ij}-\frac{\partial J}{\partial w_{ij}}=w_{ij}-\frac{\partial J}{\partial a_{j}}\frac{\partial a_j}{\partial w_{ij}}=w_{ij}+\delta_jz_i$$
	Observation: the weight changes in proportion to the input and the error at the output
	Now,
	- For *output units*, $\delta_j=(t_j-y_j)$ 
	- For hidden units, we have to take into account every unit that $j$ sends output to $$\frac{\partial J}{\partial a_{j}}=\sum_k\frac{\partial J}{\partial a_{k}}\frac{\partial a_k}{\partial a_j}$$
		To know how changing the input changes the error, the two terms in the above eq means:
		1. How the error changes as the $a_k$ (input to the next units) changes, times
		2. How they changes as current input $a_j$ changes
		We found if $j$ is a hidden unit,$$\delta_j=-\frac{\partial J}{\partial a_{j}}=g'(a_j)\sum_j\delta_kw_{jk}$$
- Thus,$$w_{ij}=w_{ij}+\alpha\ \delta_j\ z_j$$
- ## Continue on Backprop
- Derivation
	Skip
	- Backpropagation is a *linear operation*: 
		- Problem: apply too many linear operation would make the gradient become 0 or explode; 
- (Online) Backprop steps:
	1. Present a pattern, prop the activation forward till hit the output
	2. Compute the $\delta$'s at the output (usually $t-y$)
	3. Propagate the $\delta$'s backwards thru the network. Now every unit has a delta
	4. Change the weights accordingly$$w_{ij}\leftarrow w_{ij}+\alpha\delta_jz_i$$
- Why is this useful? 
	- Learns internal representations in the service of the task - learns *on its own*!
	- Efficient
	- Generalizes to *recurrent networks* (just as our brain)
- Representations
	-  In the next k slides, where k is some medium-sized integer, we will look at  
		- various representations backprop has learned, and  
		- problems backprop has solved  
	- The mantra here is:  
		- Backprop learns representations in the service of the task