import random
from collections import deque
from typing import NamedTuple

import gym
import numpy as np

import torch
from torch import nn, optim
from torch.nn import functional as F


class EpsilonGreedy:

    def __init__(self,
                 initial_epsilon: float,
                 decay_rate: float,
                 min_epsilon: float):

        self.initial = initial_epsilon
        self.decay_rate = decay_rate
        self.min = min_epsilon
        self.current = initial_epsilon

    def get(self):
        return self.current

    def decay(self):
        self.current *= self.decay_rate
        self.current = max(self.min, self.current)


class ReplaySample(NamedTuple):

    observation: torch.Tensor
    next_observation: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor


class ReplayMemory:

    def __init__(self, observation_space, max_size: int):
        self.observation = np.empty(shape=(max_size,) + observation_space.shape, dtype="float32")
        self.next_observation = np.empty(shape=(max_size,) + observation_space.shape, dtype="float32")
        self.action = np.empty(shape=(max_size,), dtype=int)
        self.reward = np.empty(shape=(max_size,), dtype="float32")
        self.dones = np.empty(shape=(max_size,), dtype=bool)
        self.max_size = max_size
        self.pointer = 0
        self.full = False

    def push(self, observation, next_observation, action, reward, done):
        self.observation[self.pointer] = observation
        self.next_observation[self.pointer] = next_observation
        self.action[self.pointer] = action
        self.reward[self.pointer] = reward
        self.dones[self.pointer] = done
        self.pointer += 1
        if not self.full:
            self.full = self.pointer >= (self.max_size - 1)
        self.pointer %= self.max_size

    def sample(self, batch_size: int) -> ReplaySample:
        if not self.full:
            raise RuntimeError("ReplayMemory must be filled before sampling.")

        indices = np.random.randint(0, self.max_size, size=batch_size)
        replay_sample = ReplaySample(
            observation=torch.from_numpy(self.observation[indices]),
            next_observation=torch.from_numpy(self.next_observation[indices]),
            action=torch.from_numpy(self.action[indices]),
            reward=torch.from_numpy(self.reward[indices]),
            done=torch.from_numpy(self.dones[indices]))
        return replay_sample


class DQN(nn.Module):

    def __init__(self,
                 observation_space,
                 action_space,
                 epsilon_greedy: EpsilonGreedy):

        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(in_features=observation_space.shape[0], out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=action_space.n))
        self.epsilon_greedy = epsilon_greedy
        self.action_space_n = action_space.n

    def forward(self, x: torch.Tensor, training: bool):
        q = self.stack.forward(x)
        action = torch.argmax(q, dim=-1)
        if training:
            epsilon = self.epsilon_greedy.get()
            if random.random() < epsilon:
                action = torch.randint(self.action_space_n, size=action.shape)
        return action

    def train_step(self, replay_sample: ReplaySample, gamma: float, optimizer):

        action_taken_mask = F.one_hot(replay_sample.action)

        self.eval()
        q_next = self.stack.forward(replay_sample.next_observation).max(dim=1).values.detach()
        bellman_target = (1. - replay_sample.done.type(torch.float32)) * q_next * gamma + replay_sample.reward
        bellman_target = bellman_target.detach()

        self.train()
        q = self.stack.forward(replay_sample.observation)
        dif = q * action_taken_mask - bellman_target[:, None] * action_taken_mask
        loss = torch.mean(torch.square(dif))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return {"loss": loss.item()}


def main():

    BATCH_SIZE = 32
    GAMMA = 0.99
    REPLAY_MEMSIZE = 1000

    env: gym.Env = gym.make("CartPole-v1")
    epsilon_greedy = EpsilonGreedy(initial_epsilon=1., decay_rate=0.9999, min_epsilon=0.1)

    dqn = DQN(env.observation_space, env.action_space, epsilon_greedy)
    optimizer = optim.Adam(dqn.parameters(), lr=1e-4)

    memory = ReplayMemory(env.observation_space, max_size=REPLAY_MEMSIZE)
    reward_buffer = deque(maxlen=100)

    for epoch in range(1, 1001):

        observation = env.reset()

        reward_sum = 0.
        losses = []

        for step in range(1, 201):

            dqn.eval()

            action = dqn.forward(torch.from_numpy(observation[None, ...]), training=True)[0].item()
            next_observation, reward, done, info = env.step(action)

            memory.push(observation, next_observation, action, reward, done)

            observation = next_observation

            reward_sum += reward

            if memory.full:
                batch = memory.sample(batch_size=BATCH_SIZE)

                dqn.train()
                report = dqn.train_step(batch, GAMMA, optimizer)

                losses.append(report["loss"])

                epsilon_greedy.decay()

            if done:
                break

        reward_buffer.append(reward_sum)

        print(f"\r [*] Epoch {epoch}/1000 Reward: {np.mean(reward_buffer):.0f}"
              f" Loss: {np.mean(losses):.4f}"
              f" Eps: {epsilon_greedy.current:.4f}", end="")
        if epoch % 100 == 0:
            print()
        if np.mean(reward_buffer) >= 150:
            print(" [*] DQN converged.")
            return


if __name__ == '__main__':
    main()
