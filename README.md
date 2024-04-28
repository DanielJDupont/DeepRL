# Deep Reinforcement Learning

### About Daniel Dupont:

#### I am making a commitment to writing a comprehensive resource on Deep Reinforcement Learning. Two of my most significant daily hobbies include learning new languages for travel and hypertrophy training for mountaineering.


> ***Seek freedom and become captive. Seek discipline and become free.***

#### Why Should You Learn Deep Reinforcement Learning?

There is nothing more interesting than watching these agents make superhuman-like decisions to accomplish their goals.


> ***I am only passionately curious.***



#### What Does This Resource Hope to Accomplish?

To improve my personal understanding of Deep Reinforcement Learning.

> ***A person needs new experiences. Without change, something sleeps inside us, and seldom awakens.***

---

### Introduction:

#### Deep Reinforcement Learning (DRL) algorithms learn directly from interacting with an environment, using neural networks with reinforcement to adapt to complex situations. These algorithms excel in settings such as games and simulations with decisions. The learning process is driven by rewards rather than traditional labeled datasets.

#### These algorithms are autonomous and can operate in novel situations very efficiently.

#### These algorithms are limited:
- The agent tends to exploit the environment in unintended ways.
- Training is expensive and time-consuming.
- They do not generalize well outside of the environment in which they have been trained.
- Human oversight is required for the design of the algorithms (neural network design, hyperparameter tuning, reward function design).

---

## Algorithms Covered:
- **Deep Q-Network (DQN)**:
DQN uses deep neural networks to approximate Q-values for action selection, stabilizing learning with techniques like experience replay and fixed Q-targets.

- **Double Deep Q-Network (Double DQN)**:
Double DQN improves upon the original DQN by reducing overestimation of Q-values through the use of two separate networks to decouple the selection and evaluation of actions.

- **Dueling Deep Q-Network (Dueling DQN)**:
Dueling DQN enhances DQN by using a network architecture that separately estimates state values and the advantages of each action, improving estimation accuracy.

- **Proximal Policy Optimization (PPO)**:
PPO is a policy gradient method that facilitates stable training via a novel objective function with clipped probability ratios, optimizing policy adjustments without large performance swings.

- **Asynchronous Advantage Actor-Critic (A3C)**:
A3C accelerates and diversifies learning by running multiple agent-environment instances in parallel, each updating a global network asynchronously to improve stability and convergence.

- **Monte Carlo Tree Search (MCTS) with DRL**:
MCTS with DRL integrates the strategic depth of Monte Carlo Tree Search with deep neural networks to enhance decision-making capabilities in complex environments like board games.

- **Soft Actor-Critic (SAC)**:
SAC is an off-policy actor-critic algorithm that optimizes a stochastic policy in continuous action spaces by maximizing a trade-off between expected return and entropy, a measure of randomness in action selection.

- **Trust Region Policy Optimization (TRPO)**:
TRPO ensures robust policy improvement by employing a trust region in policy space to manage updates, thus preventing disruptive steps during learning.

- **Rainbow**:
Rainbow unifies several enhancements to DQN, including double Q-learning, prioritized experience replay, and dueling networks to significantly boost performance and learning stability.

