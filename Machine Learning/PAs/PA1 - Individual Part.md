## Problems from Bishop
Given$$S_d=\frac{2\pi^{d/2}}{\Gamma(d/2)} \tag{1.43}$$
- ### Problem 1.1
	Let radius $r=1$ as assuming unit sphere, then (1.4.3) reduces to the well-known expressions for 
		- dimention $d=2: S_d=\frac{2\pi}{1}=2*\pi*1=2\pi r$, which the perimeter of circle
		- dimention $d=3: S_d=\frac{2\pi^{3/2}}{\pi^{1/2}/2}=4*\pi^{3/2-1/2}*1^2=4\pi r^2$, which is the surface area of sphere

- ### Problem 1.2 
	- Derive the volume of a hypersphere of radius $a$ in $d$-dimensions using (1.4.3)
		We [know](https://piazza.com/class/lcppgrpt8e441a/post/15) that the surface area of a hypersphere with radius $r$ in $d$-dimensions is $S_d(r)=S_d\ r^{d-1}$,
		Thus the volume of hypersphere with radius $a$ can be computed as $$V_d=\int_0^aS_d(r)\ dr=S_d\int_0^ar^{d-1}\ dr=S_d(\frac{a^d}{d}-\frac{0^d}{d})=\frac{S_da^d}{d} \tag{1.45}$$
	- Derive the ratio of the hypersphere volume $V_d$ to a hypercube volume $V_d'$ of side $2a$
		We get$$\frac{V_d}{V_d'}=\frac{\frac{S_da^d}{d}}{(2a)^d}=\frac{\frac{2a^d\pi^{d/2}}{\Gamma(d/2)}}{2^da^dd}=\frac{\frac{\pi^{d/2}}{\Gamma(d/2)}}{2^{d-1}d}=\frac{\pi^{d/2}}{d2^{d-1}\Gamma(d/2)} \tag{1.46}$$
	- Show the ratio goes to zero when $d\to\infty$
		Assume $x$ is large, we can use Stirling's approximation $$\Gamma(x+1)\simeq(2\pi)^{1/2}e^{-x}x^{x+1/2} \tag{1.47}$$
		Then we get $$
		\begin{align}
		\lim_{d\to\infty} \frac{V_d}{V_d'}
		&=\lim_{d\to\infty} \frac {\pi^{d/2}} {d2^{d-1} (2\pi)^{\frac{1}{2}}e^{-\frac{d}{2}-1}(\frac{d}{2})^{\frac{d-1}{2}}} \\ 
		&=\lim_{d\to\infty} 2^{\frac{1}{2}-d+\frac{d}{2}-\frac{1}{2}} \pi^{\frac{d-1}{2}} e^{1+\frac{d}{2}} d^{\frac{-1-d}{2}} \\
		&=\lim_{d\to\infty} 2^{-\frac{d}{2}}  \pi^{\frac{d-1}{2}} e^{1+\frac{d}{2}} d^{\frac{-1-d}{2}}
		\end{align}$$
		As tetration dominate all other exponential terms and $\lim_{d\to\infty}d^{\frac{-1-d}{2}}=0$, the ratio goes to zero
	- Show the ratio of the distance from the center of the hypercube to one of the corners divided by the perpendicular distance to one of the faces is $\sqrt{d}$ 
		The distance from the center $\vec c=(c_1, c_2, ..., c_d)$ to any corner $\vec p=(p_1, p_2, ..., p_d)$ can be computed as the Euclidean distance $d_{corner}=\sqrt{\sum^d_{i=1} (c_i-p_i)^2}=\sqrt{\sum^d_{i=1} a^2}=\sqrt{da^2}=a\sqrt{d}$
		The perpendicular distance from the center to one of the faces is half side length $2a$, i.e. $a$
		Thus, the ratio becomes $\frac{a\sqrt{d}}{a}=\sqrt{d}$ 
		
- ### Problem 1.3 		
	- Show the fraction of the volume of the sphere which lies at values of the radius between $a-\epsilon$ and $a$, where $0<\epsilon<a$ $$f=\frac{V_{d,a}-V_{d,a-\epsilon}}{V_{d,a}}=\frac{S_d(a^d-(a-\epsilon)^d)}{d}/\frac{S_d(a^d)}{d}=\frac{a^d-(a-\epsilon)^d}{a^d}=1-(1-\frac{\epsilon}{a})^d$$
	- Evaluate the limit$$\lim_{d\to\infty} 1-(1-\frac{\epsilon}{a})^d =1-\lim_{d\to\infty} (1-\frac{\epsilon}{a})^d$$ 
		Notice that, since we assume $a>\epsilon>0$, $1>1-\frac{\epsilon}{a}>0$, so $\lim_{d\to\infty} (1-\frac{\epsilon}{a})^d=0$ and$$\lim_{d\to\infty} 1-(1-\frac{\epsilon}{a})^d =1$$
	- Evaluate at given values
		$\epsilon/a=0.01$ and 
				- $d=2:f=1-(1-0.01)^2=0.0199$
				- $d=10:f=1-(1-0.01)^{10}=0.095617925$
				- $d=1000:f=1-(1-0.01)^{1000}=0.999956829$
		$\epsilon=a/2$ and
				- $d=2:f=1-(1-0.5)^2=0.75$
				- $d=10:f=1-(1-0.5)^{10}=0.999023438$
				- $d=1000:f=1-(1-0.5)^{1000}\approx 1$ 

- ### Problem 1.4
	- Show that 
		1) for the probability function $p(x)$ in $d$ dimensions $$p(\text{x})=\frac{1}{(2\pi\sigma^2)^{d/2}} \exp(-\frac{||\text{x}||^2)}{2\sigma^2})$$changing variables from Cartesian to polar coordinates results in a probability mass inside a thin shell of radius $r$ and thickness $\epsilon$, $\rho(r)\epsilon$, where$$\rho(r)=\frac{S_dr^{d-1}}{(2\pi\sigma^2)^{d/2}} \exp(-\frac{r^2}{2\sigma^2})$$
		2) $\rho(r)$ has a single maximum which, for large $d$, is located at $\hat r \simeq \sqrt{d}\sigma$ 	
		3) by considering $\rho(\hat r+\epsilon)$ where $\epsilon<<\hat r$ show for large $d$ $$\rho(\hat r+\epsilon)=\rho(\hat r)\exp(-\frac{\epsilon^2}{\sigma^2})$$
		1) Changing the coordinate from $\text{x}$ to $r,\epsilon$, we have 
		$$\begin{align}
		 \int^\infty_{-\infty}...\int^\infty_{-\infty}p(x_1, ...,x_d)\ dx_1...dx_d \\
		(\text{By 1.42}) &= S_d \int^{\infty}_0p(r)r^{d-1}\ dr \\
		&= S_d \int^\infty_0 \frac{1}{(2\pi\sigma^2)^{d/2}} \exp(-\frac{r^2}{2\sigma^2})r^{d-1}\ dr \\
		&= \int^\infty_0 \frac{S_d}{(2\pi\sigma^2)^{d/2}}\exp(-\frac{r^2}{2\sigma^2})r^{d-1}\ dr\\
		&= \int^\infty_0 \rho(r)\ r^{d-1}\ dr\\
		(\text{Integral over prob\. function}) &= 1
		\end{align}
		$$
		Let $\epsilon=r^{d-1}$, we then show the wanted probability mass function
		1) 
			$$\rho'(r)=\frac{S_d(d-1)}{(2\pi\sigma^2)^{d/2}\ }r^{d-2} \exp(-\frac{r^2}{2\sigma^2}) - \frac{S_d}{(2\pi\sigma^2)^{d/2}\ 2\sigma^2}r^{d-1} \exp(-\frac{r^2}{2\sigma^2})2r
			$$
			
		Set to 0 and we have $$
			\begin{align}
			\frac{S_d(d-1)}{(2\pi\sigma^2)^{d/2}\ }r^{d-2} \exp(-\frac{r^2}{2\sigma^2}) &= 
			\frac{S_d}{(2\pi\sigma^2)^{d/2}\ \sigma^2}r^{d} \exp(-\frac{r^2}{2\sigma^2}) \\
			d-1 &= 
			\frac{1}{\sigma^2}r^2 \\
			\hat r &= 
			\sqrt{d-1}\ \sigma 
			(\text{Assume large d})\ \hat r &\simeq \sqrt{d}\ \sigma
			
			\end{align}
			$$
		3) 
		Let $c=\frac{S_d}{(2\pi\sigma^2)^{d/2}}$ then $\rho(r)=cr^{d-1}\exp(-\frac{r^2}{2\sigma^2})$, and$$
			\begin{align}
			\rho(\hat r+\epsilon)
			&= c(\hat r+\epsilon)^{d-1}\exp(-\frac{(\hat 
				r+\epsilon)^2}{2\sigma^2})\\
			&= c\exp((d-1)\ln(\hat r+\epsilon)-\frac{(\hat 
				r+\epsilon)^2}{2\sigma^2}) \\
			&= c\exp(-\frac{r^2}{2\sigma^2}+(d-1)\ln\hat r-
				\frac{2r\epsilon+\epsilon^2}{2\sigma^2}+(d-1)\ln(1-\frac{\epsilon}{\hat r})) \\
			&= cr^{d-1}\exp(-\frac{r^2}{2\sigma^2})\exp(-
				\frac{2r\epsilon+\epsilon^2}{2\sigma^2}+(d-1)\ln(1-\frac{\epsilon}{\hat r})) \\
			&= \rho(\hat r)\exp(-
				\frac{2r\epsilon+\epsilon^2}{2\sigma^2}+(d-1)\ln(1-\frac{\epsilon}{\hat r}))
				
			\end{align}
			$$
			Thus, need to show in given conditions,$$\exp(-\frac{3\epsilon^2}{2\sigma^2})\approx
			\exp(-\frac{2r\epsilon+\epsilon^2}{2\sigma^2}+(d-1)\ln(1-\frac{\epsilon}{\hat r})) \tag{1.4.1}$$
			By Taylor seriesm, since $\epsilon<<\hat r$, we have $$\ln(1-\epsilon/\hat r)\approx\epsilon/\hat r-(\epsilon/\hat r)^2/2
			=\epsilon/\hat r-\epsilon^2/2\hat r^2$$
			Then $$
			\begin{align}
			(1.4.1)
			&\approx
				\exp(-\frac{2r\epsilon+\epsilon^2}{2\sigma^2}+(d-1)(\epsilon/\hat r-\epsilon^2/2\hat r^2))\\
			(d \text{ is large}) \\
			&\approx
				\exp(-\frac{2r\epsilon+\epsilon^2}{2\sigma^2}+d(\epsilon/\hat r-\epsilon^2/2\hat r^2))\\
			(\hat r \simeq \sqrt{d}\sigma) 
			&=
			\exp(- \hat r\epsilon/\sigma^2 
				 - \epsilon^2/2\sigma^2 
				 + \hat r\epsilon/\sigma^2
				 - \epsilon^2/2\sigma^2) \\
			&=
			\exp(-\epsilon^2/\sigma^2)
			\end{align}$$

- ### Problem 2
	Assume weight vector in the first quadrant, i.e. $w_0 >= 0$
	Let $\vec w=(w_1,...,w_d)$ be the weight vector except for the bias $w_0$
	Let $\vec x=(x_1,...,x_d)$ be any point lies on the boundary
	Thus, the distance from the decision boundary to the origin can be written as $$
	l
	=||proj_\vec w\vec x||
	=||\vec x||\cos(\vec w,\vec x)
	=||\vec x||\cdot\frac{\vec w\cdot\vec x}{||\vec w||\cdot||\vec x||}
	=\frac{\vec w\cdot\vec x}{||\vec w||}$$
	Also notice that for any point lies on the decision boundary, $$\vec w \cdot \vec x + w_0\cdot x_0=\vec w \cdot \vec x + w_0=0$$
	Thus by replacing $\vec w \cdot \vec x$ we have$$l=\frac{-w_0}{||\vec w||}$$ 

- ### Problem 3
	First, the gradient of the logistic activation function$$
	\begin{align}
	\frac{\partial g(\vec w^T\vec x)}{\partial w_j}
	&= -(1+e^{-x})^{-2}\cdot-e^{-x}\cdot\frac{\partial g(\vec w^T\vec x)}{\partial w_j} \\
	&= \frac{e^{-x}}{(1+e^{-x})^2}\cdot x_j \\
	&= \frac{1+e^{-x}}{(1+e^{-x})^2}
		- \frac{1}{(1+e^{-x})^2}
		\cdot
		x_j \\
	&= \frac{1}{(1+e^{-x})^2}
		\cdot
		(1-\frac{1}{(1+e^{-x})^2})
		\cdot 
		x_j \\
	&= g(\vec w^T\vec x)(1-g(\vec w^T\vec x))
		\cdot
		x_j 
	\end{align}$$
	Then take the derivative of the Cross-Entropy cost function$$
	\begin{align}
	\frac{\partial E(\vec w)}{\partial w_j}
	&=
		-\sum_{n=1}^N(
			\frac{t^n}{y^n}
				\cdot
				\frac{\partial g(\vec w^T\vec x^n)}{\partial w_j}
				\cdot
			-\frac{1-t^n}{1-y^n}
				\cdot
				\frac{\partial g(\vec w^T\vec x^n)}{\partial w_j}
			) \\
	&= -\sum_{n=1}^N(
			\frac{t^n}{y^n}
				\cdot
				g(\vec w^T\vec x^n)(1-g(\vec w^T\vec x^n))
				\cdot
				x_j 
			-\frac{1-t^n}{1-y^n}
				\cdot
				g(\vec w^T\vec x^n)(1-g(\vec w^T\vec x^n))
				\cdot
				x_j 
			) \\
	(y^n=g(\vec w^T\vec x^n))
	&= -\sum^N_{n=1}(t^n-t^ny^n-y^n+t^ny^n)x_j^n \\
	&= -\sum^N_{n=1}(t^n-y^n)x_j^n
	\end{align}
	$$