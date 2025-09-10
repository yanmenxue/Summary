## Monta Carlo
$\int_{p(x)} f(x) = \frac{\sum_{i=1}^n f(x_i)}{n}$ if we sample x_i, i = 1,2,...,n from p(x) \\
if we sample $x_i$ from p_1(x), we can estimate $E_{x \perl p_2(x)} f(x)$ as follows: \\
$E_{x \sim p_2(x)} f(x) = \int p_2(x) f(x) dx = \int p_1(x) \frac{p_2(x)}{p_1(x)} f(x) dx = E_{x \sim p_1(x)}\frac{p_2(x)}{p_1(x)} f(x)$ \\
so in ppo or grpo, we use data from old policy to optimize new policy, and we need the item $\frac{p_{new}(x)}{p{old}(x)}$  \\

# What is the target for training?

We need a target value $V_{\text{target}}(s_t)$ to train our network against. The best target we can use is the actual return we experienced from state $s_t$, which we can compute from the trajectory:

$$\hat{V}(s_t) = \sum_{l=0}^{T-t} \gamma^l r_{t+l}$$

However, this is a Monte Carlo estimate and can have high variance. A more common and stable approach is to use bootstrapping. We use our current value network to estimate the future and combine it with the immediate reward. This is the TD-target:

$$V_{\text{target}}(s_t) = r_t + \gamma V(s_{t+1})$$

In practice, for stability, we often use a mix of these ideas, like the n-step return:

$$V_{\text{target}}(s_t) = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \ ... \ + \gamma^n V(s_{t+n})$$

## Training Loop:

1.  **Collect** a trajectory of experiences $(s_t, a_t, r_t, s_{t+1})$.
2.  For each state $s_t$, **compute the target** $V_{\text{target}}(s_t)$ using one of the methods above.
3.  **Update** the value network parameters $\phi$ to minimize the loss (usually Mean Squared Error):
    $$L(\phi) = \frac{1}{N}\sum_t (V_{\text{target}}(s_t) - V_{\phi}(s_t))^2$$

Your value network's prediction $V_{\phi}(s_t)$ is your estimate for $V(s_t)$.
