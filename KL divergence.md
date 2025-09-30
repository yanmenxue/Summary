# KL 散度定义 #

$K L[q, p]=\sum_x q(x) \log \frac{q(x)}{p(x)}=\mathbb{E}_{x \sim q}\left[\log \frac{q(x)}{p(x)}\right]$

# 蒙特卡洛估计 #

最直接的蒙特卡洛估计器是基于KL散度的定义：

$$
k_1=\log \frac{q(x)}{p(x)}=-\log r, \quad \text { 其中 } \quad r=\frac{p(x)}{q(x)}
$$


这个估计器 $k_1$ 是无偏的，即其期望等于真实的KL散度。然而，由于 $\log r$ 的值在正负之间变化 （当 $(r>1)$ 时为正，当 $(r<1)$ 时为负），其方差较高，而KL散度本身始终为正。这种高方差使得（ $k_1$ ）在实际应用中表现不佳。

# 低方差的偏倚估计器 #

Schulman提出了一种替代估计器：

$$k_2=\frac{1}{2}\left(\log \frac{p(x)}{q(x)}\right)^2=\frac{1}{2}(\log r)^2$$


这个估计器 $\left(k_2\right)$ 虽然有偏，但方差显著低于 $\left(k_1\right)$ 。其优点在于：

1．始终为正：每个样本都反映了 $(p)$ 和 $(q)$ 之间的差异，且结果非负，与KL散度的性质一致。

2．低偏倚：$k_2$ 的期望是一个f－散度，其形式为：

$$
\mathbb{E}_q\left[k_2\right]=\mathbb{E}_q\left[\frac{1}{2}(\log r)^2\right]
$$


# 无偏且低方差的估计器 #

为了兼顾无偏和低方差，Schulman引入了控制变量（control variate）方法。利用（ $\mathbb{E}_q[r-1]=0$ ），可以构造一个新的估计器：

$$
k_3=-\log r+\lambda(r-1)
$$


通过选择适当的 $(\lambda)$ ，可以降低方差。当 $(\lambda=1)$ 时，估计器变为：

$$
k_3=(r-1)-\log r
$$


由于对数的凹性，（ $\log r \leq r-1$ ），因此（ $k_3$ ）始终为正。这个估计器不仅无偏，而且方差低于$k_1$。实验表明，当真实 KL 散度为 0.5 时，$k_3$ 的标准差为真实值的1．7倍，低于 $k_2$ 的1．73倍，且无偏。

# DeepSeek GRPO中的应用#

DeepSeek的GRPO算法直接采用了Schulman提出的无偏估计器（ $k_3$ ）。具体来说，GRPO使用以下估计器来近似 $D_{K L}\left(\pi_\theta \| \pi_{r e f}\right)$:

$$
D_{K L}\left(\pi_\theta \| \pi_{ref}\right)=\frac{\pi_{ref}\left(o_{i, t} \mid q, o_{i,<t}\right)}{\pi_\theta\left(o_{i, t} \mid q, o_{i,<t}\right)}
-\log \frac{\pi_{r e f}\left(o_{i, t} \mid q, o_{i,<t}\right)}{\pi_\theta\left(o_{i, t} \mid q, o_{i,<t}\right)}-1
$$


其中，$(\phi_\theta)$ 表示策略网络，$\left(\phi_{r e f}\right)$ 表示参考策略，$\left(o_{i, t}\right)$ 表示在时间 $(t)$ 的观测，$(q)$ 和 $\left(o_{i,<t}\right.$ ）分别表示上下文和历史观测。令 $\left(r=\frac{\pi_{r e f}\left(o_{i, t} \mid q, o_{i,<t}\right)}{\pi_\theta\left(o_{i, t} \mid q, o_{i,<t}\right)}\right)$ ，该估计器可重写为：

$$
k_3=(r-1)-\log r
$$



