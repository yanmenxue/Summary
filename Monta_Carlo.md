## Monta Carlo
$\int_{p(x)} f(x) dx = \frac{\sum_{i=1}^n f(x_i)}{n}$ if we sample $x_i$, i = 1,2,...,n from p(x) 

if we sample $x_i$ from $p_1(x)$, we can estimate $E_{x \sim p_2(x)} f(x)$ as follows:  

$E_{x \sim p_2(x)} f(x) = \int p_2(x) f(x) dx = \int p_1(x) \frac{p_2(x)}{p_1(x)} f(x) dx = E_{x \sim p_1(x)}\frac{p_2(x)}{p_1(x)} f(x)$ 

so in ppo or grpo, we use data from old policy to optimize new policy, and we need the item $\frac{p_{new}(x)}{p{old}(x)}$  

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

# 2. Computing the Action-Value Function $Q(s_t, a_t)$

$Q^{\pi}(s_t, a_t)$ is the expected total discounted reward from taking action $a_t$ in state $s_t$ and then following policy $\pi$ thereafter.

In Actor-Critic methods like PPO, we almost never directly estimate a Q-value network. Instead, we express $Q(s_t, a_t)$ in terms of $V(s_t)$ and the advantage.

This is based on a fundamental identity. The Q-value can be thought of as the value of the state plus the extra advantage gained from taking a specific action:

$$Q^{\pi}(s_t, a_t) = A^{\pi}(s_t, a_t) + V^{\pi}(s_t)$$

So, if we have a good estimate for $V(s_t)$ (from our value network) and a good way to estimate the advantage $A(s_t, a_t)$ (e.g., using GAE), we can implicitly get the Q-value.

## Why is this preferred?
Estimating a single value function $V(s)$ is simpler and more data-efficient than estimating a Q-function $Q(s, a)$ for every possible action in every state. This is a key reason for the success of Actor-Critic methods.

## Practical Workflow in PPO (with GAE)
Here is the complete, step-by-step process for how $V$ and the implied $Q$ are computed and used in a standard PPO implementation:

1.  **Collect Data:** Roll out the current policy for N steps (e.g., 2048 steps in a game), recording states, actions, rewards, and next-states.
    `trajectory = (s_0, a_0, r_0, s_1, s_1, a_1, r_1, s_2, ..., s_{T-1}, a_{T-1}, r_{T-1}, s_T)`

2.  **Compute Value Estimates:** Use the current value network $V_{\phi}$ to estimate the value for every state in the trajectory.
    `values = [V(s_0), V(s_1), V(s_2), ..., V(s_T)]`

3.  **Compute TD-Errors ($\delta_t$):** For each timestep t, calculate the TD-error. This is a key building block.
    $$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

4.  **Compute Advantages ($A_t$) with GAE:** Use the TD-errors from the whole trajectory to compute the advantage for each timestep. This is the GAE formula:
    $$A_t = \sum_{l=0}^{k} (\gamma \lambda)^l \delta_{t+l}$$
    (In practice, this is computed efficiently with a backward pass.)

5.  **Compute "Q-Targets" / Returns ($\hat{R}_t$):** Now that you have advantages $A_t$ and state values $V(s_t)$, you can compute the target for the value function. This target is effectively the empirical Q-value.
    $$\hat{R}_t = A_t + V(s_t)$$
    This $\hat{R}_t$ is your best empirical estimate of $Q(s_t, a_t)$.

6.  **Update the Value Network (Critic):** Train $V_{\phi}$ to get better at predicting these empirical returns $\hat{R}_t$. This improves your estimate of V(s) for the next iteration


$$L_{critic}(\phi) = \frac{1}{N}\sum_t (\hat{R}_t - V_{\phi}(s_t))^2$$

7.  **Update the Policy Network (Actor):** Use the advantages $A_t$ to update the policy, telling it which actions were good (positive advantage) and which were bad (negative advantage). The PPO clipping objective ensures this update is stable.
