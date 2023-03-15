# ReZero is All You Need: Fast Convergence at Large Depth
Deep networks have expressive power that scales exponentially with depth
	![[Pasted image 20230310195022.png]]
Deeper networks are harder to train:
 Assuming depth $L$, if the magnitude of a perturbation is changed by a factor $r$ in each layer, both signal and gradients vanish or explode at a rate of $r^L$

## Recent finding
(using Mean Field Theory) shows how signals propagate in deep networks has found interesting properties:
	Depending on the Jacobian of the network (the slope of $F^L(x)$): the residual part of the network, **the cosine of 2 input vectors will converge to 0 (orthogonal) or 1 (aligned)!**
- If fixed point is 1: the network is stable and every input maps to the same output, so the gradient vanishes
- If fixed point is 0: the network is chaotic and similar inputs map to very different outputs, leading to exploding gradient
- Ideally, we want to init the network to be at the **edge of chaos**

### Dynamical Isometry
$$J_{io}=\frac{\partial x_L}{\partial x_O}$$
Here $x_L$ is the output of the network, $x_0$ is the input. The mean squared singular values of $J_{io}$ determine the growth or decay of the average signal as it moves through the deep network. When it approximates 1, the average signal strength is neither enhanced or attenuated.

**Dynamical Isometry** (strong condition): All singular values of $J$ must be close to 1

#### Problem with Initialization
Not all architectures can satisfy this condition
- ReLUs, Self-attention, etc.
In practice, we can use normalization instead to "fix" the issue
- BatchNorm doesn't work with the sequential data
- LayerNorm can work, but with problem
- They incur computational cose
NEED A SIMPLER WAY!

### ReZero: Residual with Zero Initialization
$$x_{i+1}=x_i+\alpha_iF[W_i](x_i)$$
Initialize this learned scaler $\alpha_i$ to zero
- Trivially satisfies dynamical isometry
- Train as deep as you want, and much faster!
![[Pasted image 20230310200522.png]]
#### Why faster?
Consider a toy linear residual network that is $L$ layers deep
$$
\begin{align}
x_L    &= (1+\alpha w)^Lx_0 \\
J_{io} &= (1+\alpha w)^L
\end{align}
$$
When $\alpha=1$: The network is sensitive to the input. You need a learning rate that is exponentially small in depth 

When $\alpha=1$: The input signal is preserved

#### Performance on CIFAR-10
![[Pasted image 20230310201206.png]]

#### Apply to Transformers
- The layer norm of transformers do not satisfy dynamical isometry
- So is the soft attention!
- ReZero solves this
![[Pasted image 20230310201605.png]]

## Summary
- Really simple way to get faster convergence
- Can train very deep networks
- Flexible: works on many architectures, no need for complex init scheme

## Future work:
- Further explore the dynamics of residual weight
- Progressively grow a ReZero network
- Explore generalization