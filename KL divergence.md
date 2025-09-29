$K L[q, p]=\sum_x q(x) \log \frac{q(x)}{p(x)}=\mathbb{E}_{x \sim q}\left[\log \frac{q(x)}{p(x)}\right]$

最直接的蒙特卡洛估计器是基于KL散度的定义：

$$
k_1=\log \frac{q(x)}{p(x)}=-\log r, \quad \text { 其中 } \quad r=\frac{p(x)}{q(x)}
$$


这个估计器 $\left(k_1\right)$ 是无偏的，即其期望等于真实的KL散度。然而，由于 $(\log r)$ 的值在正负之间变化 （当 $(r>1)$ 时为正，当 $(r<1)$ 时为负），其方差较高，而KL散度本身始终为正。这种高方差使得（ $k_1$ ）在实际应用中表现不佳。
