## Introduction
- This course (cse251b) has been mostly about *supervised learning*
- We've also briefly visited *unsupervised learning*
	- UL tries to model the distribution of the data: Autoencoders as an example
 - A third type of learning is ***reinforcement learning***
	- There is an agent that acts in an environment
	- Receives only scalar rewards, that tell it whether what it did was good or bad - sometimes very delayed!

### The Agent-Environment Interface
The variables can be continue!
- Environment produces state(s) $s$ out of a set of states, and reward $r$ at time step $t$
- Produces some actions $a$ at time step $t+1$ out of a set of possible actions, based on the state(s) and reward 
- Want to maximizing the long-term expected reward
	Let $r_t$ be the reward received at time $t$ (could be 0), the goal is to maximize 
	$$R_t=r_{t+1}+\gamma\ r_{t+2}+\gamma^2\ r_{t+3}+\dots=\sum^\infty_k\gamma^kr_{t+k+1}$$
	where, $0\leq\gamma\leq1$ is the **discount rate.** This ensures the expected reward converges.
	- shortsighted $0\leftarrow\gamma\rightarrow 1$ longsighted  
![[Pasted image 20230305151221.png]]

### What does the agent learn? A *policy*
- A policy $\pi$ is a function from states to action probabilities $$\pi(s)=P(a_1,a_2,...,a_n|s)$$
- Reinforcement learning methods specify how the agent changes its policy as a result of experience.
- Basically, we want to raise the probability of action $a_i$ when it was a good move, and lower it if it was a bad move

### In summary, policy gradient is:
1. At each step of play, sample from the softmax distribution at the output:
	$$\pi(s)=(0.1, 0.02, 0.6, ..., 0.01)$$
	$\pi$ is the network, mapping from states to probabilities of actions
2. Treat the sample as the "teacher":
	$$t=(0,0,1,0,...,0)$$ for that state/action piart
3. Compute the weight change, add it to a running average of the weight changes
4. At the end, multiply the weight changes by the sign of the reward
5. Change the weights after one game
- This will make the network increase/decrease the probabilities of **all** of the actions when it wins/loses
- Over time, good actions will become more likely

### Summary
Reinforcement learning differs from supervised learning and unsupervised learning:
- There is no specific target for the outputs of the network
- There is only a scalar reinforcement signal
- The network's own actions affect its environment
- The network must learn through trial and error
- Single networks can exceed human performance on lots of game, altho low performance on games requires *prior knowledge*
- At *this point*, no single network can play more than one game -> This means every new game has to be learned from scratch

### The Markov Property
- By $s_t$ we mean whatever info is avaliable to $a_t$ about its environment
- The sate can include immediate "sensations" and structure built up over time from sequences of sensations
- Ideally, a state should summarize past sensations so as to retain all "essential" information, i.e. it should have the **Markov Property**
	$$\text{Pr}\{s_{t+1}=s',r_{t+1}=r|s_t, a_t, r_t, \cdots, r_1, s_0, a_0\}=\text{Pr}\{s_{t+1}=s',r_{t+1}=r|s_t,a_t\}$$
	- That is, *agent only needs to know, at any time, the current state*

### Model-based and Model-free RL
- *Model-based* means the agent either has available to it, or it learns, a model of *its environment*
	- A model simply consists of a probability distribution over what will happen next:
		$P(s,a,s'):$ the probability of in state $s$, taking action $a$ leads to $s'$ 
	- Many proctical approaches to reinforcement learning are *Model-free*
- Many practical approaches are *Model-free*, for example, the **Q-learning**
- But, if have a good model, one can simulate environment and *plan*

### The Value Function
- One can learn the *value of a state*:
	- $V^\pi(s) =$ the expected long-term return of being in this state, following policy $\pi$
	- As a result, $\pi(s)=\text{argmax}_a\sum_{s'}P(s,a,s')V(s')$ 
		But this ignores exploration, which can be done by *epsilon-greedy*: With probability of $1-\epsilon$ one takes the best action, else pick a randon action
#### An Action-Value Function
- Q-learning corresponds to learning a state-action function:
	- $Q(s,a) =$ the value of taking $a$ in state $s$ when following the optimal policy (value = expected long term return)
		$$q_\pi(s,a)=E_\pi[R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\dots|S_t=s,A_t=a]$$
	- The big thing about Q-learning is that you *don't need a world model*: It tells you what to do in each state

### Q-Learing
- Q-learning corresponds to learning a state-action function:
	![[Pasted image 20230306114228.png]]
- It uses an estimate to update an estimate!! 
- Q-values will propagate back from the goal state using *temperal* difference 

### Game playing: A use of the Value function
- Playing games are like tree-search (e.g. csc384 in UofT). If the search space is too large, you use a *value function* to decide what branch to search under
- In game playing, you also need to consider the opponent's action
#### The Minimax Algorithm
- Initial state: Board position and whose turn
- Op: Legal moves
- Terminal test: A test of the state if over
- Value function: numeric value for outcome of game (leaf nodes)
	- E.g. +1 win, -1 loss; # of points scored
- **Assumption:** Opponent always choose their maximum-value move
Consider two players: Max wants largest outcome, Min wants smallest outcome
- Best Achievable payoff against best play by min 
- Perfect play for deterministic, perfect-informative game
- In reallife, the branching factor is too large.
	- So the best thing to do is to **learn a mapping** from states to values
#### TD-Gammon
(By IBM) The first (old) successful application of learning the value function - three layer net
$$w_{t+1}-w_t=\alpha(Y_{t+1}-Y_t)\sum^t_{k=1}\lambda^{t-k}\nabla_wY_k$$
	Here $Y$ is the expected value of this board position - this will be minimized when $Y_{t+1}-Y_t=0$
- This is TD-$\lambda:\lambda$ is a parameter controlling the temporal credit assignment of how much of an error detected at a given time step feeds back to correct previous estimates
	- $\lambda:=0$ no feedback beyond the current time step
	- $\lambda:=1$ error feeds back without decay in time
	- $0<\lambda<1$ smooth interpolate
#### Alpha-Go
##### 1. Train two policy nets by *supervised training* on expert positions
- One is shallow and fast, but not as good;
- The other one is deep and accurate
- The first one is used for rollouts, the second is used to train another net by policy gradient methods
	- After supervised training, the deep policy net is improved by playing a younger self many times
- ##### Policy Gradient
	Directly improve the policy
	$$\Delta w=\frac{\partial t \ln(y)}{\partial w}\text{sgn}(r)$$
	Bascially reward winning moves ($r=1$) and discourage losing moves ($r=-1$)
##### 2. The Value Network
The 3rd net is trained to predict the winner from every state
- This is supervised task: From *every state seen during play*, train the net to predict the eventual winner
- After this, the net rate a particular board position as likely to lead the a win/loss
- The difference from TD-Gammon: no "temporal difference" is used - samples can be randomized easily
##### 3. At runtime: Monte Carlo Tree Search
- The tree is traversed down to a depth limit $L$
- Actions are selected according to the maximum value of that move (Q-value), plus an "Upper Confidence Bound" - an exploration term that guarantees every action has probability of being taken > 0
- The exploration term increases the more a move is not tried; vise versa
- The node is evaluated two ways:
	1. By the value net
	2. By a rollout of the fast policy net to the end of the game
	- The scores from these are averaged and propageted up the tree
#### AlphaGo Zero: *No human knowledge* to learn to play!
The key ideas are:
1. One net for both value and policy - it is forced to learn one representation that serves both tasks
2. A simplified version of Monte-Carlo tree search (MCTS) is used to improve the policy
	1. Choose a node to expand by $Q+U:$ $Q$-value + exploration term
	2. Run it thru the net to get move probabilities
	3. Propagate values up the tree
		- Used to estimate the $Q$-value
		- And keep track of how frequently this node is visited
		- Each node in the tree keeps:
			- An estimated $Q$-value: 
			$$Q(s,a)=\frac{1}{N(s,a)\sum_{s'|s,a\rightarrow s'}V(s')}$$
				where $s,a\rightarrow s'$ means the search under this node eventually led to state $s'$ after taking $a$ from $s$
			- The number of times this action has been tried, $N(s,a)$
	4. Repeat 1600 times!
3. Use a resnet with batch normalization
During training, for each episode:
1. The lookahead search got better
2. Then the network got better
3. The search got better
4. ... Eventually the value outputs got better

## Summary
- Reinforcement learning is unlike any other machine learning approach
- The system learns by acting in an environment and learns by trialds and error
- Over time, it improves performance, and can play against itself to get better
- Combining new algorithms with deep learning are leading to systems that are much better than the best humans