GANs are a form of unsupervised learning where the model can *generate data from a distribution*
- Recall Autoencoders: given a set of data, learn a mapping from X to X through a narrow channel of hidden units
- ==Why autoencoder use SSE? I understand that linear regression use SSE as the loss function but isn't there other losses?==
### How to do nonlinear dimensionality reduction
- Nonlinear representation (aka the "data manifold") can be learned by having more than one hidden layer

### [GAN](https://arxiv.org/abs/1406.2661)
Like autoencoder, but no input (no encoder), just the decoder (aka the *generator network*). There is also an *adversary*: the *discriminator network*.
The goal is to develop a generator network that learns the *distribution of the training data* - where we specify the hidden distribution: random noise!
- The generator tries to fool the discriminator with its output image with real image
- The discriminator (pretrained, fixed) tries to determine the true real image, and give the decision to the generator
- #### We can specify the objective function as a minimax expression
	$$\min_{\theta_{g}}\max_{\theta_{d}}[E_{x\sim p(data)}\log D_{\theta_d}(x)+E_{z\sim p(z)}\log(1-D_{\theta_d}(G_{\theta_g}(z)))]$$
	where $D_{\theta_d}(x)$ is the discriminator output for read data, and $D_{\theta_d}(G_{\theta_g}(z))$ is its output for generator's output
	- The discriminator can get way faster convergence than the generator. If the generator generates bad sample, it doesn't train well, because when the distriminator's output is near 0 (logistic function), gradient is tiny -> bad!
		![[Pasted image 20230306205426.png]]
- Instead, we have another objective: Gradient ascent on the discriminator output, i.e., $\max_{\theta_g}\log D_{\theta_d}(G_{\theta_g}(z))$ 
	![[Pasted image 20230306205449.png]]
- #### Training procedures:
	1. Give the generator random noise input, which leads to something (usually junk, initially) as output, repeat $k$ times
	2. Take $k$ real samples from the training set
	3. Train the discriminator to discriminate real from fake
	4. Give the generator random noise input, repeat $k$ times
	5. Fix the discriminator, and train on the generator objective function
	6. Rinse, repeat
	- This is still tricky to optimize - the discriminator get way ahead of the generator, giving little feedback ([Recent work](https://arxiv.org/abs/1701.07875) has mostly fixed this)
	![[Pasted image 20230306210442.png]]

### [CycleGAN](https://arxiv.org/abs/1703.10593)
Able to translate from one domain (X) to another (Y), without pairing the data!
![[Pasted image 20230306211114.png]]
- The loss is called "cycle consistency loss" - if X translates to Y, then Y should translate back to X

### ReCycle GAN
- The goal: map *videos* to *videos*
- The new video should maintain the *style* of the target
- One could generate the new video frame-by-frame using CycleGAN but they find that this approach has some problems
- Problems:
	- Perceptual mode collapse: network inputs/outputs faces with tiny (pixel-wise) differences
	- The use of the spatial information alone in 2D images makes it hard to learn the *movement style* of a particular domain: *Stylistic information* requires temporal knowledge as well
- #### Consider the cycle loss (of their version):
	$$L_c(G_X,G_Y)=\sum||x_t-G_X(G_Y(x_t))||^2$$
	In order to encode *temporal* information, they came up with *Recycle loss:*
	$$L_r(G_X,G_Y,P_Y)=\sum||x_{t+1}-G_X(P_Y(G_Y(x_{1:t})))||^2$$
	Here $P_Y$ is a *predictor net*: it predicts the next frame of video. It is trying to predict the next frame in the target domain, then $G_X$ brings it back to the source domain.
	So temporal information is encoded by trying to generate the correct next frame in the original domain
- #### ReCycle GAN
	For the predictor network, they use a U-net architecture to predict the next frame from the previous two frames
	U-net is a multi-scale, fully convolutional encoder-decoder network with skip connections to preserve resolution. It is often used for semantic segmentation