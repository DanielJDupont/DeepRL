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

To learn to apply these techniques to solve small scale systems to better find and understand near perfect decision making. Applications mainly to relatively simple games to be pragmatic, but still interesting.

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

### Basic Function Categories:

There are learned functions and given functions. In machine learning, functions are learned from data rather than given. For example, Newton's second law of motion $F = ma$ is a given function with a known relationship between force, mass, and acceleration. In contrast, a learned function like a line of best fit is determined by analyzing data points.

An example of a given deterministic function:

$$
f(x) = 2x + 3
$$

There are deterministic functions and probabilistic / stochastic functions. Functions in machine learning may be probabilistic, giving different outputs for the same input. Deterministic functions give the same consistent output for the same repeated input.

An example of a given probabilistic function where there is a 50% chance to select one of two actions:

$$
f(a) = \begin{cases}
0.5 & \text{if } a = a_1 \\
0.5 & \text{if } a = a_2
\end{cases}
$$

Functions in machine learning may be learned and probabilistic.

The goal of reinforcement learning is to learn the policy function that will suggest the best possible action for every situation that will lead to the maximum long term reward. The policy function can be deterministic or probabilistic. If the policy function is probabilistic, the policy function will suggest the best possible probabilities with which each action should be chosen in any given situation.

An example of a learned determinstic function where $\beta_0$ and $\beta_1$ are calculated by some other function based on given data:

$$
f(x) = \beta_0 + \beta_1 x
$$

An example of a learned probabilistic function, the logistic regression equation, it has only two possible outcomes $y = 0$ and $y = 1$. It makes no sense to attempt to calculate more than two outcomes for this equation ($y = 2$, $y= 3$, ...). The division $\frac{1}{1 + e^{-(\beta_0 + \beta_1 x)}}$is used to ensure an output between 0 and 1. $\beta_0$ and $\beta_1$ are learned, calculated by some other function based on data. $P$ is the final result, the probability, a number between 0 and 1. By convention we only calculate $y = 1$ since it's obvious what $y = 0$ will be. A value such as 40 health points for $x$ can be subbed into the equation, which may yield a probability of 0.731 or a 73.1% chance of $y$ being 1 as opposed to 0. This could mean a 73.1% chance to attack ($y = 1$) or a 26.9% chance to defend ($y = 0$) if you are at 40 hitpoints:

$$
P(y = 1 | x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x)}}
$$

A more general learned probabilistic equation, the softmax function for multi-class classification, the nuances of the equation are not important right now:

$$
P(y = k | \mathbf{x}) = \frac{e^{\mathbf{w}_k^T \mathbf{x} + b_k}}{\sum_{j=1}^K e^{\mathbf{w}_j^T \mathbf{x} + b_j}}
$$

The above equation is available in PyTorch simply as:

```python
.softmax(x)
```

If there are four possible actions, the output can be a Python list of values of probabilities to take each action in a given situation $x$. A given situation or state might be as simple as just the number of hitpoints we have.

```python
## Attack, Defend, Heal, Flee
[0.10, 0.15, 0.50, 0.25]
```

### Basic Function Notation:

There are common patterns to the notation used in deep reinforcement learning. There are variations and differences from author to author.

**States**: (I like to think simplistically of a state as the situation you are in, in a game, at a given time) typically denoted as **$s$**. But some authors use **$x$** if they work a lot in control theory. If there are discrete timesteps (there often are), some authors may use **$S_t$** for the state at time **$t$**. **$s_0$** might be the state the agent finds itself in, with other states labelled as **$s_1$**, **$s_2$**, ... , etc. This explicit time labelling of states isn't necessary.

**Actions**: are typically denoted as **$a$** but the control theorists often denote actions as **$u$**. Actions may also be time labelled **$A_t$**. We could generally write the individual actions as **$a_n^t$** as three different actions available at timestep 0: **$a_0^0$**, **$a_1^0$**, **$a_2^0$**. Perhaps these actions change every timestep in a game with these actions being very different: **$a_0^1$**, **$a_1^1$**, **$a_2^1$**. This notation isn't necessary and it just overcomplicates things.

**Value Function and Q-Function**: This is typically denoted at **$V(s)$** indicating how much reward you can expect in the long run for being in the current situation or state you are currently in. **$Q(s, a)$** indicates for your current situation, how much long term reward can you expect from each of your avaiable actions. Some authors labels these as **$V^π(s)$** and **$Q^π(s, a)$** but this is unnecessary. Some authors indicate that there is time involved with **$V_t(s)$** and **$Q_t(s, a)$** but this is also unnecessary. Many authors use the expectation operator $\mathbb{E}$ but this isn't necessary.

$$
V^\pi(s) = \mathbb{E}_{\pi}[G_t \mid S_t = s] = \mathbb{E}_{\pi} \left[\sum_{k=0}^\infty \gamma^k R_{t+k+1} \mid S_t = s\right]
$$

We are interested in understanding the concepts and applying this technology. We are not interested in dabbling in elaborate mathematical proofs.

This be more easily understood with unnecessary symbols removed and changed, with this essentially meaning the same thing as the above, without the plethora of symbols explicity reinstating what is already obvious or implied:

$$
V(s) = \sum_{t=0}^\infty \gamma^t R_{t+1}
$$

**Policy Notation**: The function that the agent uses to make decisions or the policy function or simply the policy is denoted as **$π$**. Written more verbosely as **$π(a∣s)$** which represents the probabilies of taking each action for a single given state of $s$. An input of $s$ will generate some list of probabilities for every available $a$. Deterministic policy functions may be labelled as **$μ(s)$** by some authors, where there is only one $a$ output possible for a given $s$. Some authors use **$π_θ(a∣s)$** to indicate that there are values of θ altering the function π, but this is also unnecessary.

**Reward or Return**: Typically denoted with $r$ or $R$, some choose to use $R_t$ to be explicit with time, which is not needed. Some use $G$ and $G_t$ instead of $R$. The discount factor in reward calculations is typically denoted as $γ$ but some use $β$ or other symbols.

### Work in Progress:

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
