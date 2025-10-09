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

1．软状态价值函数（Soft State－Value Function） $V\left(s_t\right)$ ：
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


其中： $r=R\left(s, a, s^{\prime}\right)$

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
\tilde{a}_t=f_left(\epsilon_t ; s_t\right)
$$


其中 $\epsilon_t$ 是来自某个固定分布（如高斯分布）的噪声。通过这种参数化，损失函数可以近似为：

$$
J_\pi(\phi)=\mathbb{E}_{s_t \sim \mathcal{D}, \epsilon_t \sim \mathcal{N}}\left[\alpha \log \pi_\phi\left(f_\phi\left(\epsilon_t ; s_t\right) \mid s_t\right)-\min _{i=1,2} Q_{\theta_i}\left(s_t, f_\phi\left(\epsilon_t ; s_t\right)\right)\right]
$$

注意，我们不直接从 $\pi_\phi(|s_t)$ 中采样 $a_t$, 而是分离随机性，从一个简单的与 $\phi$ 无关的分布中采样（如正态分布），然后，我们通过一个由 $\phi$ 参数化的、确定性的、可微的函数，将状态 $s_t$ 和噪声 $\epsilon_t$ 映射为最终的动作 $a_t$。

直观理解：这个损失函数鼓励策略 $\pi_\phi$ 选择能使得 $Q$ 值最大化的动作（第二项），但同时要保证自身的熵足够大（第一项，因为 $\log \pi$ 与熵直接相关）。

推导： 

$$
D_{K L}\left(\pi_\phi \| \pi_{\text {target }}\right)=\mathbb{E}_{a \sim \pi_\phi\left(\cdot \mid s_t\right)}\left[\log \pi_\phi\left(a \mid s_t\right)-\log \pi_{\text {target }}\left(a \mid s_t\right)\right]
$$


将我们的目标分布 $\pi_{\text {target }}=\frac{\exp \left(\frac{1}{\alpha} Q_\theta\left(s_t, a\right)\right)}{Z_\theta\left(s_t\right)}$ 代入：

$$
\begin{aligned}
D_{K L} & =\mathbb{E}_{a \sim \pi_\phi}\left[\log \pi_\phi\left(a \mid s_t\right)-\log \left(\frac{\exp \left(\frac{1}{\alpha} Q_\theta\left(s_t, a\right)\right)}{Z_\theta\left(s_t\right)}\right)\right] \\
& =\mathbb{E}_{a \sim \pi_\phi}\left[\log \pi_\phi\left(a \mid s_t\right)-\left(\frac{1}{\alpha} Q_\theta\left(s_t, a\right)-\log Z_\theta\left(s_t\right)\right)\right] \\
& =\mathbb{E}_{a \sim \pi_\phi}\left[\log \pi_\phi\left(a \mid s_t\right)-\frac{1}{\alpha} Q_\theta\left(s_t, a\right)+\log Z_\theta\left(s_t\right)\right]
\end{aligned}
$$


2．分离与策略参数 $\phi$ 相关的项
现在我们得到了KL散度的完整表达式：

$$
D_{K L}=\mathbb{E}_{a \sim \pi_\phi}\left[\log \pi_\phi\left(a \mid s_t\right)\right]-\frac{1}{\alpha} \mathbb{E}_{a \sim \pi_\phi}\left[Q_\theta\left(s_t, a\right)\right]+\log Z_\theta\left(s_t\right)
$$


C．温度参数 $\alpha$ 的自动调节
手动调节温度参数 $\alpha$ 很困难。SAC 通常将其设定为一个可优化的目标，以使得策略的平均熵维持在一个目标值 $\overline{\mathcal{H}}$（通常是 $-dim(\mathcal{A})$ ，即动作维度的负数）附近。

我们通过最小化关于 $\alpha$ 的损失函数来实现：

$$
J(\alpha)=\mathbb{E}_{a_t \sim \pi_t}\left[-\alpha\left(\log \pi_t\left(a_t \mid s_t\right)+\overline{\mathcal{H}}\right)\right]
$$


在实践中，我们使用：

$$
J(\alpha)=\mathbb{E}_{s_t \sim \mathcal{D}}\left[-\alpha\left(\log \pi_\phi\left(a_t \mid s_t\right)+\overline{\mathcal{H}}\right)\right]
$$


如果策略的熵低于目标值，这个损失会降低 $\alpha$ ，减弱熵的权重；反之则会提高 $\alpha$ 。
