### 1. Maximum Likelihood Estimation
$$
	\begin{align}
	p(x_1, \dots, x_n;\lambda) \\
	(\text{since }x_1,\dots,x_n\ i.i.d)&= \prod_{i=1}^n p(x_i;\lambda) \tag{1} \\
	&= \lambda^n\exp(-\lambda\sum_{i=1}^nx_i) \\
	\end{align}	
$$
Then the log-likelihood can be shown as
$$
\begin{align}
l(x_1, \dots, x_n;\lambda) &= \ln p(x_1, \dots, x_n;\lambda) \\
&= n\ln\lambda-\lambda\sum_{i=1}^nx_i \\
\end{align}	
$$
Then set its derivative to equal to 0 and get the MLE estimator $$
	\begin{align}
	\frac{\partial l}{\partial\lambda} 
	&= \frac{n}{\lambda}-\sum_{i=1}^nx_i=0 \\
	\Rightarrow \frac{n}{\hat\lambda}&=\sum_{i=1}^nx_i \\
	\hat\lambda&=\frac{n}{\sum_{i=1}^nx_i}
	\end{align}
	$$
### 2
#### 2.1 Derivation
##### 2.1.1 Derivation of $\delta_k^n$:
$$
\begin{align}
\delta_k^n 
&= -\frac{\partial E^n}{\partial a_k^n} \\
&= -\sum_j\frac{\partial E^n}{\partial y_j^n}\frac{\partial y^n_j}{\partial a_k^n} \tag{2}
\end{align}$$
We first derive the derivative w.r.t. $a_k^n$ of the softmax function $y^n_j=\frac{\exp a_j^n}{\sum_{k'}\exp a_{k'}^n}$
Let $\sum_{k'}a^n_{k'}=\sum$ . There are two cases to consider: 
$$
\begin{align}
\text{when }j=k:\frac{\partial y^n_k}{\partial a_k^n} \\
(\text{By the quotient rule}) &= \frac{(\exp(a_k^n))'\sum-\exp(a_k^n)(\sum)'}{\sum^2}\\
&= \frac{\exp(a_k^n)\sum-\exp(a_k^n)\cdot\exp(a_k^n)}{\sum^2} \\
&= \frac{\exp a_k^n}{\sum}\cdot\frac{\sum-\exp a_k^n}{\sum} \\
&= y^n_k(1-y^n_k) \tag{3} 
\end{align}
$$
$$
\begin{align}
\text{when }j\neq k:\frac{\partial y^n_j}{\partial a_k^n} \\
(\text{By the quotient rule}) &= \frac{(\exp(a_j^n))'\sum-\exp(a_j^n)(\sum)'}{\sum^2}\\
&= \frac{0-\exp(a_j^n)\exp(a_k^n)}{\sum^2} \\
&= -\frac{\exp a_j^n}{\sum}\cdot\frac{\exp a_k^n}{\sum} \\
&= -y^n_jy^n_k \tag{4}
\end{align}
$$
We then can compute the delta $$
\begin{align}
(2) &=-\sum_{i=1}^Kt_i^n\frac{\partial\ln y_i^n}{\partial y_i^n}\frac{\partial y^n_i}{\partial a_k^n} \\
&= -\sum_{i=1}^K\frac{t_i^n}{y^n_i}\frac{\partial y^n_i}{\partial a_k^n} \\
(\text{By 3, 4}) &= \sum_{i\neq k}t_i^ny^n_k-t_k^n(1-y_k^n) \\
&= y^n_k\sum_{i\neq k}t_i^n-t_k^n+t_k^ny_k^n \\ 
&=y^n_k\sum^K_{i=1}t_i^n-t^n_k \tag{5}
\end{align}$$
Since we assume examples to belong to one and only one category, we have 
$$
(5)=y^n_k\cdot1-t^n_k=y^n_k-t^n_k \tag{6}
$$
==Opposite result?==
##### 2.1.2 Derivation of $\delta_j^n$
We first derive the derivative of $\tanh$ w.r.t. $a_j^n$ 
$$
\begin{align}
\frac{\partial a_k^n}{\partial a_j^n}
&= \frac{\partial\tanh(a_j^n)}{\partial a_j^n}\\
(\text{By the quotient rule})
&= \frac{(e^{a_j^n}-e^{-a_j^n})(e^{a_j^n}+e^{-a_j^n}) - (e^{a_j^n}-e^{-a_j^n})(e^{a_j^n}-e^{-a_j^n})}{(e^{a_j^n}+e^{-a_j^n})^2} \\
&= 1- \frac{(e^{a_j^n}-e^{-a_j^n})^2}{(e^{a_j^n}+e^{-a_j^n})^2} \\
&= 1-\tanh^2(a_j^n) \tag{7}
\end{align}
$$
Similar as $\delta_k^n$ we can derive
$$
\begin{align}
\delta_j^n
&= -\sum_k\frac{\partial E^n}{\partial a_k^n}\frac{\partial a^n_k}{\partial a_j^n}\\
&= -\sum_{k=1}^K(y_k^n-t_k^n)\frac{\partial a^n_k}{\partial a_j^n} \\
(\text{By 7})&= -\sum_{k=1}^K(y_k^n-t_k^n)(1-\tanh^2(a_j^n)) \\
&= (1-\tanh^2(a_j^n))\sum_{k=1}^K(t_k^n-y_k^n) \tag{8}
\end{align}
$$
==Where is the w_ij==

#### 2.2 Update rule
##### 2.2.1 Update rule for the hidden layer $w_{jk}$
$$
\begin{align}
w_{jk}
&= w_{jk}-\alpha\frac{\partial E}{\partial w_{jk}} \\
&= w_{jk}-\alpha\sum_n\frac{\partial E^n}{\partial w_{jk}} \\
&= w_{jk}-\alpha\sum_n\frac{\partial E^n}{\partial a_k^n}\frac{\partial a_k^n}{\partial w_{jk}} \\
(\text{By 6})&= w_{jk}-\alpha\sum_n(y^n_k-t^n_k)x_j^n
\end{align}$$
##### 2.2.2 Update rule for the hidden layer $w_{ij}$
$$
\begin{align}
w_{ij}
&= w_{ij}-\alpha\frac{\partial E}{\partial w_{ij}} \\
&= w_{ij}-\alpha\sum_n\frac{\partial E^n}{\partial w_{ij}} \\
&= w_{ij}-\alpha\sum_n\frac{\partial E^n}{\partial a_j^n}\frac{\partial a_j^n}{\partial w_{ij}} \\
(\text{By 7})&= w_{ij}-\alpha\sum_n[(1-\tanh^2a_j^n)\sum_{k=1}^K(t_k^n-y_k^n)x_{i}^n]
\end{align}
$$
#### 2.3 Vectorize computation
##### 2.3.1 Update rule for the hidden layer $w_{jk}$ in matrix form
$$
\begin{align}
W_{K\times J}=W_{K\times J}-\alpha(Y_{K\times 1}-T_{K\times 1})\cdot X_{J\times 1}^T
\end{align}
$$
##### 2.3.2 Update rule for the hidden layer $w_{ij}$ in matrix form
$$
\begin{align}
W_{J\times I}=W_{J\times I} 
-\alpha(1-\tanh^2(W_{J\times I}
\cdot X_{I\times 1}^T))
\cdot [(T_{K\times 1}-Y_{K\times 1})
\cdot X_{I\times 1}^T]
\end{align}
$$
