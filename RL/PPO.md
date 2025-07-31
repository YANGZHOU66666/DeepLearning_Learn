# PPO

## RL基础

**Action Space:** 可选择的动作，比如 {left, up, right}

**Policy:** 策略函数，输入 State，输出 Action 的概率分布。一般用 π 表示。

π(left|sₜ) = 0.1
π(up|sₜ) = 0.2
π(right|sₜ) = 0.7

**Trajectory:** 轨迹，用 τ 表示，一连串状态和动作的序列。Episode,Rollout。 {s₀, a₀, s₁, a₁, …}

sₜ₊₁ = f (sₜ, aₜ) 确定
sₜ₊₁ = P (・|sₜ, aₜ) 随机（比如游戏里开宝箱，给定“开宝箱”动作，由于宝箱内物品随机，下一个状态也是随机的）

**Return:** 回报，从当前时间点到游戏结束的 Reward 的累积和。



强化学习的目标：训练一个Policy神经网络$\pi$，在所有状态S下，给出相应的Action，得到Return的期望最大

强化学习的目标：训练一个Policy神经网络$\pi$，在所有的Trajectory中，得到Return的期望最大



## 公式推导

要优化的式子：$E_{τ∼P_θ(τ)}[R(τ)]=∑_τR(τ)P_θ(τ)$，其中$τ$是某条轨迹，这个式子表示策略下所有被采样轨迹的期望return

使用梯度上升法，两边同时取梯度得：（要优化的参数是$\theta$）

$\nabla E_{τ∼P_θ(τ)}[R(τ)]=\nabla \sum_τR(τ)P_θ(τ)=\nabla\sum_\tau R(\tau)\nabla P_\theta(\tau)=\sum_\tau R(\tau)\nabla P_\theta(\tau)\frac{P(\tau)}{P(\tau)}=\sum_\tau R(\tau) P_\theta(\tau)\frac{\nabla P(\tau)}{P(\tau)}$

这里$\sum_\tau R(\tau) P_\theta(\tau)$可以通过采样n份求平均来近似预估，因此：

$原式\approx \frac{1}{N}\sum_{i=1}^N R_\theta(\tau_i)·\frac{\nabla P_\theta(\tau_i)}{P_\theta(\tau_i)}=\frac{1}{N}\sum_{i=1}^N R(\tau_i)·\nabla log(P_\theta(\tau_i))$

现在考虑怎么表示$P_\theta(\tau_i)$。我们知道一条Trajectory是由初始状态与一系列action导致的状态转移导致的，因此只需要对各个状态取走这条路径的action概率连乘即可：

$P_\theta(\tau_i)=\Pi_{t=1}^{T_i}P_\theta(a_i^t|s_i^t)$，其中$T_i$表示第i条路径的长度，$P_\theta(a_i^t|s_i^t)$表示第i条路径上，第t个状态取转化到第(t+1)个状态的动作的概率。

因此：

$上式=\frac{1}{N}\sum_{i=1}^N R(\tau_i)·\nabla log(\Pi_{t=1}^{T_i}P_\theta(a_i^t|s_i^t))=\frac{1}{N}\sum_{i=1}^N R(\tau_i)· \sum_{t=1}^{T_i}\nabla log(P_\theta(a_i^t|s_i^t))=\frac{1}{N}\sum_{i=1}^{N}\sum_{t=1}^{T_i}R(\tau_i)·\nabla log(P_\theta(a_i^t|s_i^t))$

用这个梯度来做梯度上升，乘以学习率更新神经网络中的参数：Policy Gradient

第一部分：Trajectory得到的return，第二部分是
