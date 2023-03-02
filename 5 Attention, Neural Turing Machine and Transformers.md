
(This part is heavily based on [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/#born-for-translation))
# Attention [Bahdanau et al., 2015](https://arxiv.org/pdf/1409.0473.pdf)
- *First came to serve seq2seq on neural machine translation (NMT) in 2015 but soon migrated to FNN, creating the transformers!
- Instead of building a single fix-length context vector out of the encoder's last hidden state, attention creates *shortcuts*, with customizable weights, between the context vector and the entire source input => Avoid forgetting!

## Definition: Let's define in more formal way
![[Pasted image 20230225220752.png]]
	Say, we have a source sequence $\mathbf{x}$ of length $n$ and try to output a target sequence $\mathbf{y}$ of length $m$:
	$$\mathbf{x}=[x_1, x_2, ..., x_n]$$
	$$\mathbf{y}=[y_1, y_2, ..., y_m]$$
- The encoder is a bidirectional RNN (in the original paper) with :
	- An encoder state is a concat of the two $\mathbfit{h_i}=[\overrightarrow{\mathbfit{h_i}};\overleftarrow{\mathbfit{h_i}}]$ for $i\in\{1,...,n\}$, where a forward hidden state $\overleftarrow{\mathbfit{h_i}}$ and a backward one $\overrightarrow{\mathbfit{h_i}}$. 
- The decoder network has "
	- hidden state $\mathbfit{s_t=f(s_{t-1},y_{t-1},c_t)}$ for the output word at position $t\in\{1,...,m\}$, where 
	- the context vector $\mathbfit{c_t}$ is a sum of hidden states of the input sequence, weighted by alignment scores:
	$$
	\begin{align}
	\mathbfit{c_t} &= \sum^n_{i=1}\alpha_{t,i}\mathbfit{h_i} &&\text{; Context vector for output }y_t \\
	\alpha_{t,i}   &= \text{align}(y_t, x_i) &&\text{; How well two words are aligned} \\
				   &= \text{Softmax}(\text{score}(\mathbfit{s_{t-1},h_i})) &&\text{; Softmax of some predefined alignment score}
	\end{align} \\
	$$
	In Bahdanau’s paper, the alignment score $\alpha$ is parametrized by a FFN with a single hidden layer, and this network is jointly trained with other parts of the model. The score function is therefore in the following form, given that $\tanh$ is used as the non-linear activation function:
	$$\text{score}(\mathbfit{s_t,h_i})=\mathbf{v}_a\tanh(\mathbf{W}_a[\mathbfit{s_t;h_i}])$$
	where both $\mathbf{W}_a$ and $\mathbf{v}_a$ are weight matrices to be learned in the alignment model.
- #### Scaled Dot-Product: Another attention scoring (Used in trasnformers!)
	$$\text{score}(\mathbfit{s_t,h_i})=\frac{\mathbfit{s}_t^\top h_i}{\sqrt{n}}$$
	where the scaling factor, $n$, is the dimension of the source hidden state
- #### A nice alignment score matrix can be obtained
	Notice that in the middle, attention figures out different word orders
	![[Pasted image 20230226123651.png]]
## Summary
- Adding attention to RNNs allows them to "attend" to different parts of the input at every time step  
- The general attention layer is a new type of layer that can be used to design new neural network architectures  

## Family of Attention Mechanisms 
Full list can be found in the blog

## Self-Attention
**Self-attention**, also known as **intra-attention**, is an attention mechanism relating different positions of a **single sequence** in order to compute a representation of the same sequence. It has been shown to be very useful in machine reading, abstractive summarization, or image description generation.
	
The [long short-term memory network](https://arxiv.org/pdf/1601.06733.pdf) paper used self-attention to do machine reading. In the example below, the self-attention mechanism enables us to learn the correlation between the current words and the previous part of the sentence.
	![[Pasted image 20230226151929.png]]

## Soft vs Hard Attention
-  **Soft** Attention: the alignment weights are learned and placed “softly” over all patches in the source image; essentially the same type of attention as in [Bahdanau et al., 2015](https://arxiv.org/abs/1409.0473).
	-   _Pro_: the model is smooth and differentiable.
	-   _Con_: expensive when the source input is large.
-  **Hard** Attention: only selects one patch of the image to attend to at a time.
	-   _Pro_: less calculation at the inference time.
	-   _Con_: the model is non-differentiable and requires more complicated techniques such as variance reduction or reinforcement learning to train. ([Luong, et al., 2015](https://arxiv.org/abs/1508.04025))
## Global vs Local Attention
[Luong, et al., 2015](https://arxiv.org/pdf/1508.04025.pdf) proposed the “global” and “local” attention. The global attention is similar to the soft attention, while the local one is an interesting blend between [hard and soft](https://lilianweng.github.io/posts/2018-06-24-attention/#soft-vs-hard-attention), an improvement over the hard attention to make it differentiable: the model first predicts a single aligned position for the current target word and a window centered around the source position is then used to compute a context vector.
	![[Pasted image 20230226152351.png]]

# Neural Turing Machines
**Neural Turing Machine** (**NTM**, [Graves, Wayne & Danihelka, 2014](https://arxiv.org/abs/1410.5401)) is a model architecture for coupling a neural network with external memory storage. The memory mimics the Turing machine tape and the neural network controls the operation heads to read from or write to the tape. However, the memory in NTM is finite, and thus it probably looks more like a “Neural [von Neumann](https://en.wikipedia.org/wiki/Von_Neumann_architecture) Machine”.
	![[Pasted image 20230226154944.png]]
- The controller (any type of NN) is in charge of executing ops on the mem.
- The read/write heads are softly attending to all the mem addresses.

### Read and write
- When reading from the memory at time $t$, $\mathbf{w}_t$, an attention vector of size $N$, controls how much attention to assign to different memory locations (matrix rows). The read vector $\mathbf{r}_t$ is a weighted sum by attention intensity:
	$$\mathbf{r}_t=\sum^N_{i=1}w_t(i)\mathbf{M}_t(i), \text{ where}\sum w_t(i)=1;w_t\text{ the softmax column vector}$$ $w_t(i)$ is the $i$-th element in $\mathbf{w}_t$ and $\mathbf{M}_t(i)$ is the $i$-th row vector in the memory.
- Compute $\mathbf{w}_t$ if RNN
	Controller outputs:
	- $\mathbf{k}_t$ the key vector for content-addressable memory - we want the memory element most similar to this vector (for 3x4 mem this would be a length 4 vector)
	- $\beta_t$ the *gain parameter* on the content-match (a large $\beta$ *sharpen* the address i.e. focus more)
	- $g_t$ the *switch* (gate) between content- and location-based addressing
	- $\mathbf{s}_t$ the *shift vector* of the address
	- $\gamma_t$ the *gain parameter* on the softmax address, making it more binary
	- So, $w_t^c(i)=\text{softmax}{(\beta_t K[\mathbf{k}_t, \mathbf{M}_t(i)])}$ is based on how similar that mem element is to the key, where $K$ is the cosine similarity
		E.g. suppose $\mathbf{k}=(3,1,0,0)$  and the match score (cosine) is $0.8, 0.1, 0.1$ and $\beta=1$, then $w^c=(0.5,0.25,0.25)$

- When writing into the memory at time $t$, as inspired by the input and forget gates in LSTM, a write head first wipes off some old content according to an erase vector $\mathbf{e}_t$ and then adds new information by an add vector $\mathbf{a}_t$:
	$$
	\begin{align}
	\mathbf{\tilde m}_t(i) &= \mathbf{m}_{t-1}(i)[1-w_t(i)\mathbf{e}_t]  &&; \text{erase} \\
	\mathbf{m}_t(i) &= \mathbf{\tilde m}_t(i) + w_t(i)\mathbf{a}_t &&; \text{add}
	\end{align}
	$$
	Why two steps? Because the memory is a bunch of linear neurons. There are no *overwrite* operation
# Transformers: Attension is all you need! 
[“Attention is All you Need”](http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf) is one of the most impactful and interesting paper in 2017. It presented a lot of improvements to the *soft attention* and make it possible to do seq2seq modeling _without_ recurrent network units. The proposed “**transformer**” model is entirely built on the self-attention mechanisms without using sequence-aligned recurrent architecture.
(Nice blogs! [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/) [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/); The Stanford cs231n [slides](chrome-extension://cdonnmffkdaoajfknoeeecmchibpmkmg/assets/pdf/web/viewer.html?file=http%3A%2F%2Fcs231n.stanford.edu%2Fslides%2F2022%2Flecture_11_ruohan.pdf) and [git repo](https://github.com/cs231n/cs231n.github.io/blob/master/transformers.md))
## Transformer vs. RNN
- ### RNNs  
	- (+) LSTMs work reasonably well for long sequences.  
	- (-) Expects an ordered sequences of inputs  
	- (-) Sequential computation: subsequent hidden states can only be computed after the previous ones are done.  
- ### Transformer:  
	- (+) Good at long sequences. Each attention calculation looks at all inputs.  
	- (+) Can operate over unordered sets or ordered sequences with positional encodings.  
	- (+) Parallel computation: All alignment and attention scores for all inputs can be done in parallel.  
	- (-) Requires a lot of memory: N x M alignment and attention scalers need to be calculated and stored for a single self-attention head. (but GPUs are getting bigger and better)  

## An aside: What is attention?
When focus on a particular aspect of environment - a 'pointing' to something
- ***Overt* attention** is when *moving eyes* towards something or someone
- ***Covert* attention** is when, *without moving eyes*, directing attention to something 
### How! can we 'point' with neural networks?
- Control camera movements (overt attention)
- Direct attention to ***internal* activations: *Softmax + multiplicative connections***

## Key, Value and Query

The major component in the transoformer is the unit of **multi-head** **self-attention** mechanism. The encoded representation of the input is viewed as a set of **key-value** pairs, ($\mathbf{K,V}$), both of dimension $n$ (input sequence length). 

In the context of NMT, both are the encoder hidden states. In the decoder, the previous output is compressed into a **query** ($\mathbf{Q}$ of dimension $m$) and the next output is produced by mapping this query and the set of keys and values.

The transformer adopts the [scaled dot-product attention](https://lilianweng.github.io/posts/2018-06-24-attention/#summary) (mentioned above): the output is a weighted sum of the values, where the weight assigned to each value is determined by the dot-product of the query with all the keys:
$$
\text{Attention}(\mathbf{Q, K, V})=\mathbf{Z}=
\frac{\text{softmax}( \mathbf{QK^\top})}{\sqrt{n}}\mathbf{V}$$

- The output $\mathbf{Z}$ has the same dimension as the input. 
- The $\text{softmax}(\frac{\mathbf{QK^\top}}{\sqrt{n}})$ measures the ratio of how much *attention* we should be paying to $\mathbf{V}$.
- The ${\sqrt{n}}$ is to relief the gradient vanishing problem

## Multi-Head Self-Attention

![[Pasted image 20230226164856.png]]
- ### Self-attention
	If the $\mathbf{Q, K, V}$ are the *same* matrix $\mathbf{X}$, then it is *self-attention* mechanism!
	In transformer, we will perform learnable linear transformations on the same $\mathbf{X}$ to focus on 
	
- #### Multi-Head
 The multi-head mechanism runs independent attentions multiple times in parallel, which are simply concatenated and linearly transformed (weighted sum) into the expected dimensions. (I assume the motivation is because ensembling always helps?) 
	
 According to the paper, _“multi-head attention allows the model to jointly attend to information from different representation **subspaces** at different positions. With a single attention head, averaging inhibits this."_
	I think what that mean is to reduce the influence of initial states of Q, K, V
	$$
	\begin{align}
	\text{MultiHead}(\mathbf{Q,K,V})&=[\text{head}_1;...;\text{head}_h]\mathbf{W}^O \\
	\text{where head}_i &= \text{Attention}(\mathbf{XW}_i^Q, \mathbf{XW}_i^K, \mathbf{XW}_i^V)
	\end{align}
	$$
	where $\mathbf{W}_i^Q,\mathbf{W}_i^K,\mathbf{W}_i^V,\mathbf{W}^O$ are learnable matrices

## Encoder
![[Pasted image 20230226165659.png]]
The encoder generates an attention-based representation
- A stack of $N=6$ identical layers.
- Each layer has a **multi-head self-attention layer** and a **fully connected FFN**.
- Each sub-layer adopts a residual connection and a layer normalization. All the sub-layers output data of the same dimension $d=512$.

## Decoder
![[Pasted image 20230226185350.png]]
- Every specs are the same as encoder
- The first multi-head attention sub-layer takes in the previous output (so, decoder is sequential) is **modified** to prevent positions from attending to subsequent positions, as we don’t want to look into the future of the target sequence when predicting the current position.

## Full Architecture
![[Pasted image 20230226190139.png]]
- Both the source and target sequences first go through embedding layers to produce data of the same dimension $d=512$.
- To preserve the position information, a sinusoid-wave-based positional encoding is applied and summed with the embedding output.
- A softmax and linear layer are added to the final decoder output to classify word output.

## Summary
- Use **parallel computation** at every level of the encoder - fast to train 
- Are feed-forward
- Replicate the network for every input  
- Achieve state-of-the-art performance without recurrence!
- Transformers are a type of layer that uses self-attention and layer norm
- They are quickly replacing RNNs, LSTMs, and may(?) even replace convolutions.  
102
# ViT: Image Transformer ([ICLR 2021](https://openreview.net/forum?id=YicbFdNTTy))
