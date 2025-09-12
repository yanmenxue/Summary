# Actor-Critic 算法

## 目标函数
\[ J(\theta) = \mathbb{E}_{s \sim \rho^\pi, a \sim \pi_\theta} [R(s, a)] \]

## 策略梯度
\[ \nabla_\theta J(\theta) = \mathbb{E}_{s \sim \rho^\pi, a \sim \pi_\theta} [\nabla_\theta \log \pi_\theta(a|s) \cdot A^{\pi}(s, a)] \]
其中优势函数：
\[ A^{\pi}(s, a) = Q^{\pi}(s, a) - V^{\pi}(s) \]

## Critic 更新
\[ \delta_t = r_t + \gamma V_w(s_{t+1}) - V_w(s_t) \]
\[ \Delta w = \alpha_w \delta_t \nabla_w V_w(s_t) \]

## Actor 更新
\[ \Delta \theta = \alpha_\theta \delta_t \nabla_\theta \log \pi_\theta(a_t|s_t) \]

# SAC (Soft Actor-Critic) 算法

## 最大熵目标
\[ J(\pi) = \sum_{t=0}^T \mathbb{E}_{(s_t, a_t) \sim \rho_\pi} [r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t))] \]
其中熵项：
\[ \mathcal{H}(\pi(\cdot|s)) = -\mathbb{E}_{a \sim \pi} [\log \pi(a|s)] \]

## Soft Q-function 和 Soft V-function
\[ Q^\pi(s, a) = r(s, a) + \gamma \mathbb{E}_{s' \sim p} [V^\pi(s')] \]
\[ V^\pi(s) = \mathbb{E}_{a \sim \pi} [Q^\pi(s, a) - \alpha \log \pi(a|s)] \]

## Critic 更新
\[ J_Q(\phi) = \mathbb{E}_{(s, a, r, s') \sim \mathcal{D}} \left[ \left( Q_\phi(s, a) - \left( r + \gamma (V_{\bar{\psi}}(s')) \right) \right)^2 \right] \]
目标值：
\[ V_{\bar{\psi}}(s') = \mathbb{E}_{a' \sim \pi_\theta} [Q_{\bar{\phi}}(s', a') - \alpha \log \pi_\theta(a'|s')] \]

## Actor 更新
\[ J_\pi(\theta) = \mathbb{E}_{s \sim \mathcal{D}} \left[ \mathbb{E}_{a \sim \pi_\theta} [\alpha \log \pi_\theta(a|s) - Q_\phi(s, a)] \right] \]

## 温度参数调节
\[ J(\alpha) = \mathbb{E}_{a \sim \pi^*} [-\alpha \log \pi^*(a|s) - \alpha \mathcal{H}_0] \]
