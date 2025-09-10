## Monta Carlo
$\int_{p(x)} f(x) = \frac{\sum_{i=1}^n f(x_i)}{n}$ if we sample x_i, i = 1,2,...,n from p(x)
if we sample $x_i$ from p_1(x), we can estimate $E_{x \perl p_2(x)} f(x)$ as follows:
$E_{x \sim p_2(x)} f(x) = \int p_2(x) f(x) dx = \int p_1(x) \frac{p_2(x)}{p_1(x)} f(x) dx = E_{x \sim p_1(x)}\frac{p_2(x)}{p_1(x)} f(x)$
so in ppo or grpo, we use data from old policy to optimize new policy, and we need the item $\frac{p_{new}(x)}{p{old}(x)}$ 
