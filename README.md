# Deep Reinforcement Learning

This is a very early work in progress that I am updating on a daily basis as I read the literature. The intent is to make this resource concise and pragmatic with easy to understand applications to simple games. There will be drastic changes and updates until I figure out how to best structure this.

This resource assumes you have a full computer science undergraduate education and industry experience coding as a full stack software engineer.

---

> **_"The beginning of knowledge is the discovery of something we do not understand."_**

---

### Acknowledgements:

I extend my sincerest gratitude to ChatGPT, the esteemed professor of digital academia, for graciously co-authoring this work, whether to its benefit or its detriment.

### About Daniel Dupont:

I am making a commitment to writing a comprehensive resource on deep reinforcement learning.

Two of my most significant daily hobbies include learning new languages for travel and hypertrophy training for mountaineering.

### Suggested Tools:

Python and VSCode with these Extensions:

- Black Formatter
- Markdown Preview Enchanced

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

An example of a learned probabilistic function, the logistic regression equation, it has only two possible outcomes $y = 0$ and $y = 1$. It makes no sense to attempt to calculate more than two outcomes for this equation ($y = 2$, $y= 3$, ...). The division $\frac{1}{1 + e^{-(\beta_0 + \beta_1 x)}}$ is used to ensure an output between 0 and 1. $\beta_0$ and $\beta_1$ are learned, calculated by some other function based on data. $P$ is the final result, the probability, a number between 0 and 1. By convention we only calculate $y = 1$ since it's obvious what $y = 0$ will be. A value such as 40 health points for $x$ can be subbed into the equation, which may yield a probability of 0.731 or a 73.1% chance of $y$ being 1 as opposed to 0. This could mean a 73.1% chance to attack ($y = 1$) or a 26.9% chance to defend ($y = 0$) if you are at 40 hitpoints:

$$
P(y = 1 | x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x)}}
$$

There are more general learned probabilistic equations, like the softmax function, where we can do things like $y = 2$, $y= 3$, ... , etc. Where we can have more than two possible actions for our agent to take in a given state.

If there are four possible actions, the output can be a Python list of probabilities to take each action in a given situation $x$. A given situation or state might be as simple as just the number of hitpoints we have, where $x = 40$ might give us:

```python
## Attack, Defend, Heal, Flee
[0.10, 0.15, 0.50, 0.25]
```

### Basic Function Notation:

There are common patterns to the notation used in deep reinforcement learning. There are variations and differences from author to author.

**States**: (I like to think simplistically of a state as the situation you are in) typically denoted as **$s$**. But some authors use **$x$** if they work a lot in control theory. If there are discrete timesteps (there often are), some authors may use **$S_t$** for the state at time **$t$**. **$s_0$** might be the state the agent finds itself in, with other states labelled as **$s_1$**, **$s_2$**, ... , etc. **$s_{t+1}$** means the next state from **$s_t$**, with **$s_{t+1}$** often written as **$s_{t^′}$**.

**Actions**: are typically denoted as **$a$** but the control theorists often denote actions as **$u$**. Actions may also be time labelled **$A_t$**.

**Value Function and Q-Function**: This is typically denoted at **$V(s)$** indicating how much reward you can expect in the long run for being in the current situation or state you are currently in. **$Q(s, a)$** indicates for your current situation, how much long term reward can you expect from each of your avaiable actions. Some authors labels these as **$V^π(s)$** and **$Q^π(s, a)$**. Some authors indicate that there is time involved with **$V_t(s)$** and **$Q_t(s, a)$**. Many authors use the expectation operator $\mathbb{E}$ or $\mathbb{E}_{ \pi}$.

**Policy Notation**: The function that the agent uses to make decisions or the policy function or simply the policy is denoted as **$π$**. Written more verbosely as **$π(a∣s)$** which represents the probability of taking action **$a$** for a single given state of $s$. An input of $s$ will generate some list of probabilities for every available $a$. Deterministic policy functions may be labelled as **$μ(s)$** by some authors, where there is only one $a$ output possible for a given $s$. Some authors use **$π_θ$** or **$π_θ(a∣s)$** to indicate that there are values of θ altering the function π.

The neural network that represents **$π(a∣s)$** will output a vector that sums to 1. If we have three actions, we have three outputs for each action for a state we use as input into the neural network:
$π(a_1∣s)+π(a_2∣s)+π(a_3∣s)=1$

The policy function is state dependent with a single state used as input. A probability is output for each action. We can sample an action from the distribution of probabilities or simply take a deterministic approach and select the action with the highest probability, or use a combination of both approaches.

**Reward or Return**: Typically denoted with $r$ or $R$, some choose to use $R_t$ to be explicit with time, which is not needed. Some use $G$ and $G_t$ instead of $R$. The discount factor in reward calculations is typically denoted as $γ$ but some use $β$ or other symbols.

**Summary of Some Basic Symbols:**
**$s$** &nbsp;&nbsp;&nbsp; **$x$** &nbsp;&nbsp;&nbsp; **$S_t$** &nbsp;&nbsp;&nbsp; **$s_t$** &nbsp;&nbsp;&nbsp; **$s_{t+1}$** &nbsp;&nbsp;&nbsp; **$s_{t^′}$**
**$A$** &nbsp;&nbsp;&nbsp; **$A_t$** &nbsp;&nbsp;&nbsp; **$a$** &nbsp;&nbsp;&nbsp; **$u$**
**$π$** &nbsp;&nbsp;&nbsp; **$π_θ$** &nbsp;&nbsp;&nbsp; **$π(s)$** &nbsp;&nbsp;&nbsp; **$μ(s)$** &nbsp;&nbsp;&nbsp; **$π(a∣s)$**
**$V(s)$** &nbsp;&nbsp;&nbsp; **$V^π(s)$**
**$Q(s, a)$** &nbsp;&nbsp;&nbsp; **$Q^π(s, a)$**

### Markov Decision Processes:

#### The Markov Property Depends Entirely on the Current State and Has No Memory of Past States:

Reinforcement learning problems or sequential decision problems can be modelled as Markov decision processes (MDPs). These have the Markov propety where the next state depends on upon the current state.

Chess is an intuitive example of something that is a Markov Decision Process.

There is however a rule in chess that violates the Markov property called the **threehold repetition rule** where a game can be declared a draw if the same position occurs three times.

We can get around this by keeping some finite record of the past six or so states, in our current state. This is an approach we can use to take what are intuitively non-Markovian elements in our problems and give them the Markov property in our implementation, so we can actually solve with them.

---

> **_"A person needs new experiences. Without change, something sleeps inside us, and seldom awakens."_**

---
