A form of unsupervised learning
E.g. Human babies, Autoencoders, MAST, GANs, GPT-3, **SimCLR**, etc.
## Introduction to Contrastive Learning
- *Contrastive Learning* is an example of *metric learning*
- Metric learning is often supervised, e.g. Siamese Networks
- Metric learning refers to learning distances, used for clustering
- E.g. SMART 1.0:
	- Molecules from the same category were moved together
	- Molecules from the different categories are moved apart
### A Popular Self-supervised Learning Approach
Main idea: Learn features such that: $f_\theta(I)=f_\theta(\text{augment}(I))$
- All augmentation of the same sample map to the same place in representational space
- Invariant to "*nuisance factors*" 
- Can use other images in the minibatches as negative examples
- Unsupervised Siamese Networks

# Self-supervision in Computer Vision
  ## The promise of "alternative supervision"
  Obtain "real" labels is difficult and expensive: ImageNet with 14M images took 22 human years! Instead, obtain labels using "semi-automatic" process
  - Hashtags
  - GPS locations
  - "Self"-supervised

  ## Pretext task
  - Self-supervised learning on representation/hidden-property of the observed data 
  - Often not the "real" task (like image classification) we care about

  E.g. Given two patches of an image, predict the locations
  ## Jigsaw
  ![[Pasted image 20230310180034.png]]
  #### The hope of generalization
  We really hope the pre-training task and the transfer task are "aligned"
  E.g.
  ![[Pasted image 20230310170942.png]]
  AH! Higher layers do not generalize!
### Pre-trained features should:
- Represent how images relate to one another
- Be robust to "nuisance factors" -- invariance
AH! We can use $f_\theta(I)=f_\theta(\text{augment}(I))$
#### Problem (mentioned): Trivial Solutions
![[Pasted image 20230310171356.png]]

#### Many ways to avoid trivial solutions:
- Similarity Maximization Objective
	- Contrastive learning: SimCLR, PIRL, MoCo
	- Distillation: BYOL
- Redundancy Reduction Objective
	- Redundancy Reduction: Barlow Twins
![[Pasted image 20230310183311.png]]

## SimCLR: A Simple framework for learning Contrastive Learning of Visual Representations
- ### Idea 1: Self-supervision
	![[Pasted image 20230310154737.png]]
	1. Given image, use standard data augmentation to get two *new* images
	2. Use the images as positive example, and the rest of the batch as negative examples
	3. Typically $f()$ is ResNet-50
- ### Idea 2: Add a hidden layer on top
	![[Pasted image 20230310155027.png]]
	1. $g()$ one hidden layer network: ReLU maps to 180d representation $z$'s
	2. Apply metric learning to $z_i$
	3. Then use $h_i$ instead as representation  
		Why? 
		Conjecture: $z$ throws too much informations in $h$ - e.g. color, orientation, etc.
	- Paper tried linear/nonlinear $g()$ and no $g()$ at all. Nonlinear works the best
### Idea 3: Normalized Temperature-scaled Cross Entropy Loss
- All vectors $z$ are $l_2$ normalized to length 1
- For positive pair $(u_i,v_j)$, cosine similarity is used: $s_{ij}=\cos(u_i,v_j)$ 
- Loss is 
	$$l(i,j)=-\log\frac{\exp(s_{i,j}/\tau)}{\sum_{k=1}^{2N}\exp(\mathbb{1}_{[k\neq i]}s_{i,k}/\tau)}$$
	Top: minizie distance between a positive pair
	Bottom: maximize distance against all others
	![[Pasted image 20230310162342.png]]
	Notice:
	- Requires many samples in minibatch
	- $l(i,j)\neq l(j,i)$: not symmetric!
	- No need for negative pairs - all in the denominator
### Idea 4: 
- *Compose* transformations stochastically
- Apply cropping, random size (uniform from 0.08 to 1.0 in area)
- 3/4 to 4/3 aspect ratio change
- Then apply randomized color change, or grayscale
	Randomizing **color** is very important! Because we don't want color to be used as feature to learn
### Summary
- Use data augmentation to generate positive pairs
- Augment by randomly composing different augmentation
- Optimize on a nonlinear projection
- Use a new loss function that doesn't require *negative pairs*
	Quesiton: Why negative pairs are needed?
	-> $f_\theta(I)=f_\theta(\text{augment}(I))$ can be trivially hold by mapping everything to constant
	-> Need to learn to separate negative pairs


## Pretext-Invariance Representation Learning(PIRL)
- Group of related & non-related images $\rightarrow$ Shared networks (Siamese Net) $\rightarrow$ Image Features (Embeddings)
	![[Pasted image 20230310171825.png]]
	![[Pasted image 20230310180130.png]]

## MoCo
- Maintain "momentum" network - MoCo
- Pros - online
- Cons - extramemory for parameters/stored features, extra fwd pass compared to memory bank
![[Pasted image 20230310180606.png]]

### BYOL
Want $f_\theta(I)=f_\theta(\text{augment}(I))$
What we actually do $f_\theta^{\text{student}}(I)=f_\theta^{\text{teacher}}(\text{augment}(I))$
	Prevent trivial solutions by asymmetry
	Asymmetric learning rule / achitecture

## Borace Barlow
### Hypothesis
neurons should have **independent features** to get the most efficient representation
### Redundancy Reduction
- N neurons produce a representation: N dimensional feature
- Each neuron should satisfy
	- Invariance under different data augmentation
	- Independent of other neurons - *reduce redundancy*
	 i.e.
		$$
		\begin{align}
		f_\theta(I)[i] &=    f_\theta(\text{augment}(I))[i] \\
		f_\theta(I)[i] &\neq f_\theta(\text{augment}(I))[j] \\
		\end{align}
		$$
	![[Pasted image 20230310182444.png]]
- We compute the cross-correlation between features, and the teaching signal is the identity matrix
	![[Pasted image 20230310182604.png]]

## Masked Autoencoders are Scalable Vision Learners
### Autoencoder
Encoder-decoder model that gets input and tries to reproduce the same sample as output.
- It gains the latent representation in the middle.
- Can be trained using *self-training*
Now NLP uses deep autoencoders:
	Common technique is to mask part of the input and ask the model to reconstruct it (BERT), or to predict the next word(GPT), using the transformer achitecture
But masking image is different:
- Semantic content: words carry **more** information than image
- Images are very redundant: pixels can be reconstructed from nearby pixels
- Decoder: Simple MLP in NLP, but in CV, another transformer
#### Step 1: Break the image into non-overlapping patches
#### Step 2: Remove 75% (wow!) of the patches
#### Step 3: Encode these patches 
with their positions - otherwise it will be hard!
#### Summary
- Simple architecture, scalable
- Without negative example
- Very little data augmentation
- Less computation

