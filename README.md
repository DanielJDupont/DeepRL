# Deep Reinforcement Learning

This resource assumes you have a full computer science undergraduate education and industry experience coding as a full stack software engineer.

The beginning of knowledge is the discovery of something we do not understand.

### About Daniel Dupont:

I am making a commitment to writing a comprehensive resource on Deep Reinforcement Learning.

Two of my most significant daily hobbies include learning new languages for travel and hypertrophy training for mountaineering.

##### Why Should You Learn Deep Reinforcement Learning?

There is nothing more interesting than watching these agents make superhuman-like decisions to accomplish their goals.

##### What Does This Resource Hope to Accomplish?

To improve my personal understanding of Deep Reinforcement Learning.

---

> **_"Seek freedom and become captive. Seek discipline and become free."_**

> **_"I am only passionately curious."_**

> **_"A person needs new experiences. Without change, something sleeps inside us, and seldom awakens."_**

---

### Introduction:

Deep Reinforcement Learning (DRL) algorithms learn directly from interacting with an environment, using deep learning with reinforcement learning. These algorithms excel in settings such as games and simulations with decisions. The learning process is driven by rewards rather than traditional labeled datasets.

Deep learning is about approximating functions using neural networks.

Reinforcement learning is about learning from trial and error without pre-existing data.

Deep reinforcement learning is ideal for situations where you have something (an agent) making decisions in a complex enviroment.

Deep reinforcement learning is effective at solving sequential decision problems (or optimal control problems). Countless real world environments fit this type of problem where a sequence of decisions are made and any one bad decision can be very costly.

Puzzles and games are an easy way to learn and test these algorithms. There are many applications of Deep Reinforcement Learning in every field.

Functions in machine learning are learned functions, not given functions. Functions in machine learning may be probabilistic, giving different outputs for the same input.

The goal of reinforcement learning is to learn the policy function that will suggest the best possible action for every situation that will lead to the maximum long term reward.

#### These algorithms are autonomous and can operate in novel situations very efficiently.

#### These algorithms are limited:

- The agent tends to exploit the environment in unintended ways.
- Training is expensive and time-consuming.
- They do not generalize well outside of the environment in which they have been trained.
- Human oversight is required for the design of the algorithms (neural network design, hyperparameter tuning, reward function design).

---

## Tabular Value Based Reinforcement Learning:

- **Agent and Environment**
- **Markov Decision Processes**
- **Elements of Reinforcement Learning**

## Deep Reinforcement Learning:

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
