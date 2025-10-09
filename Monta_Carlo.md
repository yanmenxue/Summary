# Monta Carlo算法
$\int p(x) f(x) dx = E_{x \sim p(x)} f(x) = \frac{\sum_{i=1}^n f(x_i)}{n}$ (大数定律) if we sample $x_i$, i = 1,2,...,n from p(x) 

if we sample $x_i$ from $p_1(x)$, we can estimate $E_{x \sim p_2(x)} f(x)$ as follows:  

$E_{x \sim p_2(x)} f(x) = \int p_2(x) f(x) dx = \int p_1(x) \frac{p_2(x)}{p_1(x)} f(x) dx = E_{x \sim p_1(x)}\frac{p_2(x)}{p_1(x)} f(x) = \frac{\sum_{i=1}^n \frac{p_2(x_i)}{p_1(x_i)}f(x_i)}{n}$ 

so in ppo or grpo, we use data from old policy to optimize new policy, and we need the item $\frac{p_{new}(x)}{p{old}(x)}$  

# PPO算法
## 1.What is the target for training?

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

## 2. Computing the Action-Value Function $Q(s_t, a_t)$

$Q^{\pi}(s_t, a_t)$ is the expected total discounted reward from taking action $a_t$ in state $s_t$ and then following policy $\pi$ thereafter.

In Actor-Critic methods like PPO, we almost never directly estimate a Q-value network. Instead, we express $Q(s_t, a_t)$ in terms of $V(s_t)$ and the advantage.

This is based on a fundamental identity. The Q-value can be thought of as the value of the state plus the extra advantage gained from taking a specific action:

$$Q^{\pi}(s_t, a_t) = A^{\pi}(s_t, a_t) + V^{\pi}(s_t)$$

So, if we have a good estimate for $V(s_t)$ (from our value network) and a good way to estimate the advantage $A(s_t, a_t)$ (e.g., using GAE), we can implicitly get the Q-value.

## Why is this preferred?
Estimating a single value function $V(s)$ is simpler and more data-efficient than estimating a Q-function $Q(s, a)$ for every possible action in every state. This is a key reason for the success of Actor-Critic methods.

## 3.Practical Workflow in PPO (with GAE)
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

# Actor－Critic 算法
Actor－Critic 结合了策略梯度（Actor）和价值函数（Critic）两个部分。

1 主要符号
- $\pi_\theta(a \mid s)$ ：参数为 $\theta$ 的策略（Actor）
- $Q_w(s, a)$ ：参数为 $w$ 的动作价值函数（Critic）
- 状态 $s_t$ ，动作 $a_t$ ，奖励 $r_t$ ，折扣因子 $\gamma$

## 2 算法流程

### 2.1 Critic更新（用 TD 误差更新 $Q_w$ ）
TD 误差：

$$
\delta_t=r_t+\gamma Q_w\left(s_{t+1}, a_{t+1}\right)-Q_w\left(s_t, a_t\right)
$$


参数更新（梯度上升，最小化 TD 误差的平方）：

$$
w \leftarrow w+\alpha_w \delta_t \nabla_w Q_w\left(s_t, a_t\right)
$$

(使用半梯度，将 $Q_w\left(s_{t+1}, a_{t+1}\right)$ 看作是常数，不去计算这部分关于w的梯度)

在代码中通常这样处理（.detach()，这就是在告诉自动微分系统：不要计算目标值的梯度），

计算TD误差

td_target = reward + gamma * value_net(next_state).detach()

td_error = td_target - value_net(current_state)

损失函数

value_loss = 0.5 * (td_error ** 2).mean()


或者如果用的是优势函数 $A_w(s, a)=Q_w(s, a)-V_w(s)$ ，则 Critic 可能是更新 $V_w$ 。

$$
A\left(s_t, a_t\right)=r_t+\gamma V_w\left(s_{t+1}\right)-V_w\left(s_t\right)
$$


这里利用了贝尔曼方程：

$$
Q\left(s_t, a_t\right)=\mathbb{E}\left[r_t+\gamma V\left(s_{t+1}\right)\right]
$$


所以：

$$
A\left(s_t, a_t\right)=r_t+\gamma V_w\left(s_{t+1}\right)-V_w\left(s_t\right)
$$

最小化优势函数来优化 $V_w$.

### 2.2 Actor 更新（策略梯度）：

$$
\theta \leftarrow \theta+\alpha_\theta \nabla_\theta \log \pi_\theta\left(a_t \mid s_t\right) Q_w\left(s_t, a_t\right)
$$


更常见的是用优势函数 $A_w\left(s_t, a_t\right)$ 代替 $Q_w$ 来减少方差：

$$
\theta \leftarrow \theta+\alpha_\theta \nabla_\theta \log \pi_\theta\left(a_t \mid s_t\right) A_w\left(s_t, a_t\right)
$$


# Soft Actor-Critic (SAC) 算法

## 1．核心思想：最大嫡目标
标准强化学习的目标是最大化期望累积回报：

$$
J(\pi)=\mathbb{E}_{\tau \sim \pi}\left[\sum_{t=0}^{\infty} \gamma^t R\left(s_t, a_t, s_{t+1}\right)\right]
$$


SAC 在此基础上，为每一步的策略都增加一个熵项：

$$
J(\pi)=\mathbb{E}_{\tau \sim \pi}\left[\sum_{t=0}^{\infty} \gamma^t\left(R\left(s_t, a_t, s_{t+1}\right)+\alpha \mathcal{H}\left(\pi\left(\cdot \mid s_t\right)\right)\right)\right]
$$


其中：
- $\tau$ 是由策略 $\pi$ 采样得到的轨迹 $\left(s_0, a_0, s_1, a_1, \ldots\right)$ 。
- $\alpha>0$ 是一个温度参数，它决定了熵项相对于奖励的重要性。
-  $\mathcal{H}\left(\pi\left(\cdot \mid s_t\right)\right)$ 是策略在状态 $s_t$ 下的熵，定义为：

## 2．价值函数与 Q 函数
在最大熵框架下，价值函数和 Q 函数的定义也包含了末来的熵。
1．软状态价值函数（Soft State－Value Function）$V\left(s_t\right)$ ：
它衡量从状态 $s_t$ 开始，遵循策略 $\pi$ 所能得到的期望累积软回报。

$$
V\left(s_t\right)=\mathbb{E}_{\tau \sim \pi, s_0=s_t}\left[\sum_{l=t}^{\infty} \gamma^{l-t}\left(R\left(s_l, a_l, s_{l+1}\right)+\alpha \mathcal{H}\left(\pi\left(\cdot \mid s_l\right)\right)\right)\right]
$$


2．软动作价值函数（Soft Action－Value Function）$Q\left(s_t, a_t\right)$ ：
它衡量在状态 $s_t$ 执行动作 $a_t$ 后，再遵循策略 $\pi$ 所能得到的期望累积软回报。

$$
Q\left(s_t, a_t\right)=R\left(s_t, a_t, s_{t+1}\right)+\gamma \mathbb{E}_{s_{t+1}}\left[V\left(s_{t+1}\right)\right]
$$

3．价值函数与 Q 函数的关系：
将熵的定义代入 $V$ 函数，我们可以得到 $V$ 和 $Q$ 之间的关键关系：

$$
\begin{aligned}
V\left(s_t\right) & =\mathbb{E}_{a_t \sim \pi}\left[Q\left(s_t, a_t\right)-\alpha \log \pi\left(a_t \mid s_t\right)\right] \\
& =\mathbb{E}_{a_t \sim \pi}\left[Q\left(s_t, a_t\right)\right]+\alpha \mathcal{H}\left(\pi\left(\cdot \mid s_t\right)\right)
\end{aligned}
$$


这个公式是 SAC 的基石，它说明一个状态的价值等于该状态下所有动作的 Q 值的期望，再加上策略的熵。

## 3．软策略迭代（Soft Policy Iteration）

SAC 的理论基础是软策略迭代，它类似于标准的策略迭代，但适用于最大熵目标。

软策略评估（Soft Policy Evaluation）：

对于固定的策略 $\pi$ ，软 Q 函数可以通过反复应用软贝尔曼备份算子 $\mathcal{T}^\pi$ 来收敛到该策略的真

$$
\mathcal{T}^\pi Q\left(s_t, a_t\right) \triangleq R\left(s_t, a_t, s_{t+1}\right)+\gamma \mathbb{E}_{s_{t+1}}\left[V\left(s_{t+1}\right)\right]
$$


其中 $V\left(s_{t+1}\right)=\mathbb{E}_{a_{t+1} \sim \pi}\left[Q\left(s_{t+1}, a_{t+1}\right)-\alpha \log \pi\left(a_{t+1} \mid s_{t+1}\right)\right]$ 。

软策略改进（Soft Policy Improvement）：

对于给定的 Q 函数，我们通过最小化 KL 散度来更新策略，使其更接近一个以 Q 值为指数的布。

$$
\pi_{\mathrm{new}}=\arg \min _{\pi^{\prime}} D_{K L}\left(\pi^{\prime}\left(\cdot \mid s_t\right) \| \frac{\exp \left(\frac{1}{\alpha} Q^{\pi_{\mathrm{old}}}\left(s_t, \cdot\right)\right)}{Z^{\pi_{\mathrm{old}}}\left(s_t\right)}\right)
$$


其中 $Z^{\pi_{\mathrm{old}}}\left(s_t\right)$ 是一个配分函数，用于对分布进行归一化。这个更新的解析解表明，新的策略高的动作上会有更高的概率。

## 4．Soft Actor－Critic 的实践实现

在实际中，SAC 使用函数近似（神经网络）并同时学习三个网络：策略网络 $\pi_\phi$ 和两个 Q 网络 $Q_{\theta_1}, Q_{\theta_2}$ （用于缓解过高估计）。它遵循 Actor－Critic 架构。

A．软 Q 函数（Critic）的更新

我们通过最小化软贝尔曼误差来更新 Q 网络的参数 $\theta$ 。

目标 Q 值：

$$
y\left(r, s^{\prime}, d\right)=r+\gamma(1-d)\left(\min _{i=1,2} Q_{\bar{\theta}_i}\left(s^{\prime}, \tilde{a}^{\prime}\right)-\alpha \log \pi_\phi\left(\tilde{a}^{\prime} \mid s^{\prime}\right)\right), \quad \tilde{a}^{\prime} \sim \pi_\phi\left(\cdot \mid s^{\prime}\right)
$$


其中：
－$r=R\left(s, a, s^{\prime}\right)$
- $d$ 表示是否为终止状态（done flag）。
- $\tilde{a}^{\prime}$ 是从当前策略 $\pi_\phi$ 中为新状态 $s^{\prime}$ 采样的动作。
- $Q_{\bar{\theta}}$ 是目标 Q 网络，其参数 $\bar{\theta}$ 是主 Q 网络参数的指数移动平均（EMA），用于稳定训练。
损失函数：
对于两个 Q 网络 $i=1,2$ ，其损失函数为：

$$
J_Q\left(\theta_i\right)=\mathbb{E}_{\left(s, a, r, s^{\prime}, d\right) \sim \mathcal{D}}\left[\frac{1}{2}\left(Q_{\theta_i}(s, a)-y\left(r, s^{\prime}, d\right)\right)^2\right]
$$


这里 $\mathcal{D}$ 是经验回放缓冲区。

B．策略函数（Actor）的更新

策略的更新目标是最大化期望的 Q 值同时保持高熵。我们通过最小化 KL 散度来更新策略参数 $\phi$ 。
根据软策略改进的推导，策略的损失函数可以写为：

$$
J_\pi(\phi)=\mathbb{E}_{s_t \sim \mathcal{D}}\left[D_{K L}\left(\pi_\phi\left(\cdot \mid s_t\right) \| \frac{\exp \left(\frac{1}{\alpha} Q_\theta\left(s_t, \cdot\right)\right)}{Z_\theta\left(s_t\right)}\right)\right]
$$


为了便于计算，我们使用重参数化技巧（Re－parameterization Trick）。我们从策略中采样动作：

$$
\tilde{a}_t=f_\phi\left(\epsilon_t ; s_t\right)
$$


其中 $\epsilon_t$ 是来自某个固定分布（如高斯分布）的噪声。通过这种参数化，损失函数可以近似为：

$$
J_\pi(\phi)=\mathbb{E}_{s_t \sim \mathcal{D}, \epsilon_t \sim \mathcal{N}}\left[\alpha \log \pi_\phi\left(f_\phi\left(\epsilon_t ; s_t\right) \mid s_t\right)-\min _{i=1,2} Q_{\theta_i}\left(s_t, f_\phi\left(\epsilon_t ; s_t\right)\right)\right]
$$


直观理解：这个损失函数鼓励策略 $\pi_\phi$ 选择能使得 $Q$ 值最大化的动作（第二项），但同时要保证自身的熵足够大（第一项，因为 $\log \pi$ 与熵直接相关）。
