# Deep Reinforcement Learning

This resource assumes you have a full computer science undergraduate education and industry experience coding as a full stack software engineer.

---

> **_"The beginning of knowledge is the discovery of something we do not understand."_**

---

### About Daniel Dupont:

I am making a commitment to writing a comprehensive resource on deep reinforcement learning.

Two of my most significant daily hobbies include learning new languages for travel and hypertrophy training for mountaineering.

##### Why Should You Learn Deep Reinforcement Learning?

There is nothing more interesting than watching these agents make superhuman-like decisions to accomplish their goals.

##### What Does This Resource Hope to Accomplish?

To improve my personal understanding of deep reinforcement learning.

---

> **_"Seek freedom and become captive. Seek discipline and become free."_**

> **_"I am only passionately curious."_**

> **_"A person needs new experiences. Without change, something sleeps inside us, and seldom awakens."_**

---

### Introduction:

Deep reinforcement learning (DRL) algorithms learn directly from interacting with an environment, using deep learning with reinforcement learning. Deep reinforcement learning excels in settings such as games and simulations with decisions. The learning process is driven by rewards rather than traditional labeled datasets.

Deep learning is about approximating functions using neural networks.

Reinforcement learning is about learning from trial and error without pre-existing data.

Deep reinforcement learning is ideal for situations where you have something (an agent) making decisions in a complex enviroment.

Deep reinforcement learning is effective at solving sequential decision problems (or optimal control problems). Countless real world environments fit this type of problem where a series of decisions are made and any one bad decision can be very costly.

Puzzles and games are an easy way to learn and test these algorithms. There are many applications of deep reinforcement learning in every field.

There are learned functions and given functions. In machine learning, functions are learned from data rather than given. For example, Newton's second law of motion (F = ma) is a given function with a known relationship between force, mass, and acceleration. In contrast, a learned function like a line of best fit is determined by analyzing data points.

There are deterministic functions and probabilistic functions. Functions in machine learning may be probabilistic, giving different outputs for the same input. Deterministic functions give the same consistent output for the same repeated input.

Functions in machine learning may be learned and probabilistic.

The goal of reinforcement learning is to learn the policy function that will suggest the best possible action for every situation that will lead to the maximum long term reward.

These algorithms are autonomous and can operate in novel situations very efficiently.

These algorithms are limited:

- The agent tends to exploit the environment in unintended ways to maximize rewards.
- Training is expensive and time-consuming.
- They generally do not work well outside of the environment in which they have been trained.
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
