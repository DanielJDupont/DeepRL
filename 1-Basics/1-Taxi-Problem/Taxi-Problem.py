import gym
import numpy as np


def iterate_value_function(value_function_input, gamma, env):
    ret = np.zeros(env.nS)
    for sid in range(env.nS):
        temp_v = np.zeros(env.nA)
        for action in range(env.nA):
            for prob, dst_state, reward, is_final in env.P[sid][action]:
                temp_v[action] += prob * (reward + gamma * value_function_input)[dst_state] * (not is_final)
                ret[sid] = max(temp_v)
    return ret


def build_greedy_policy(value_function_input, gamma, env):
    """Build a greedy policy based on the value function."""
    new_policy = np.zeros(env.nS)

    for state_id in range(env.nS):
        rewards = np.zeros(env.nA)

        for action in range(env.nA):
            for prob, dst_state, reward, is_final in env.P[state_id][action]:
                rewards[action] += prob * (reward + gamma * value_function_input[dst_state])
            new_policy[state_id] = np.argmax(rewards)
    return new_policy


env = gym.make("Taxi-v3")
gamma = 0.9
cum_reward = 0
n_rounds = 500
env.reset()

for t_rounds in range(n_rounds):
    # init env and value function
    observation = env.reset()
    value_function = np.zeros(env.nS)

    # solve MDP
    for _ in range(100):
        old_value_function = value_function.copy()
        value_function = iterate_value_function(value_function, gamma, env)
        if np.all(value_function == old_value_function):
            break

    policy = build_greedy_policy(value_function, gamma, env).astype(np.init)

    # apply policy
    for t in range(1000):
        action = policy[observation]
        observation, reward, done, info = env.step(action)
        cum_reward += reward
        if done:
            break

    if t_rounds % 50 == 0 and t_rounds > 0:
        print(cum_reward * 1.0 / (t_rounds))


env.close()
