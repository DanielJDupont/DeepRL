# Q-Learning Basics

The "Q" in Q-learning stands for "quality". Quality refers to the long term reward of an action taken in a given situation in the environment.

The actor or agent in the environment when confronted with a situation will need to choose an action. Its decision making process is the policy function.

The agent uses the policy function to make a decision. The policy function takes in the current situation as input. The policy function will output the one best action to take, which is the action with the highest Q score. The action with the highest Q score will provide the best long term rewards for the given situation.

Basic Q learning generally is determinstic, where the agent simply uses the policy function to select which one action has the highest score compared to all other actions for a given situation.

Q learning can be made probabilsitc, the policy function can be modified so the agent selects actions probabilsitcally from a range of available actions for a situation. Actions with higher Q scores would be selected more often than actions with lower Q scores. This promotes more exploration but slows convergence towards the most optimal solution.

This is the most basic policy function. A policy function uses a $π$ symbol as opposed to an $f$ symbol to make it explicit it is for the policy or decision making of the agent. Where for a given situation or state $s$, the action $a_1$ is always output:

$$
\pi(s) = a_1
$$
