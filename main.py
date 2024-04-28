import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim


# Tic-Tac-Toe Environment
class TicTacToe:
    def __init__(self):
        self.board = np.zeros(9)
        self.game_over = False
        self.winner = None
        self.current_player = 1  # 1 for 'X', -1 for 'O'

    def is_valid_move(self, move):
        return self.board[move] == 0

    def make_move(self, move):
        if self.is_valid_move(move):
            self.board[move] = self.current_player
            self.current_player *= -1
            self.check_game_status()
            return self.board, self.reward(), self.game_over
        return self.board, 0, self.game_over

    def reward(self):
        if self.game_over:
            if self.winner == 1:
                return 1  # 'X' wins
            elif self.winner == -1:
                return -1  # 'O' wins
        return 0

    def check_game_status(self):
        lines = [
            (0, 1, 2),
            (3, 4, 5),
            (6, 7, 8),
            (0, 3, 6),
            (1, 4, 7),
            (2, 5, 8),
            (0, 4, 8),
            (2, 4, 6),
        ]
        for a, b, c in lines:
            if self.board[a] == self.board[b] == self.board[c] != 0:
                self.game_over = True
                self.winner = self.board[a]
                return
        if 0 not in self.board:
            self.game_over = True

    def reset(self):
        self.board = np.zeros(9)
        self.game_over = False
        self.winner = None
        self.current_player = 1
        return self.board


# Q-Network
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(9, 50)  # 9 inputs for the board states
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 9)  # 9 outputs for each action's Q-value

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# DQN Agent
class DQNAgent:
    def __init__(self, gamma=0.99):
        self.model = DQN()
        self.gamma = gamma  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice([i for i in range(9) if state[i] == 0])
        state = torch.tensor(state, dtype=torch.float32)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def train(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        target = reward
        if not done:
            target = reward + self.gamma * torch.max(self.model(next_state)).item()
        target_f = self.model(state)
        target_f[action] = target
        output = self.model(state)
        loss = self.criterion(output, target_f)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# Example of running the DQN on Tic-Tac-Toe
def main():
    env = TicTacToe()
    agent = DQNAgent()
    episodes = 1000

    for e in range(episodes):
        state = env.reset()
        for _ in range(9):  # max moves
            action = agent.act(state)
            next_state, reward, done = env.make_move(action)
            agent.train(state, action, reward, next_state, done)
            state = next_state
            if done:
                break


if __name__ == "__main__":
    main()
