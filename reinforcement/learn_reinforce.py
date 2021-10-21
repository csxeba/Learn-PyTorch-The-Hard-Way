from collections import deque
from statistics import mean

import gym

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F


def discount_rewards(rewards, gamma=0.99) -> torch.Tensor:
    returns = np.empty(len(rewards), dtype="float32")
    running_add = 0.
    for t in reversed(range(0, len(rewards))):
        running_add = running_add * gamma + rewards[t]
        returns[t] = running_add
    returns = (returns - returns.mean()) / returns.std()
    return torch.from_numpy(returns)


class ExperienceStore:

    def __init__(self):
        self.states = []
        self.rewards = []
        self.actions = []
        self.done = []
        self.returns = []

    def store(self, state, reward, action, done):
        self.states.append(state)
        self.rewards.append(reward)
        self.actions.append(action)
        self.done.append(done)

    def finalize(self, final_reward):
        self.returns = discount_rewards(self.rewards[1:] + [final_reward])
        self.states = torch.from_numpy(np.array(self.states))
        self.actions = torch.from_numpy(np.array(self.actions))

    def reset(self):
        self.states = []
        self.rewards = []
        self.actions = []
        self.done = []
        self.returns = []


class Policy(nn.Module):

    def __init__(self, observation_space, action_space):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(in_features=observation_space.shape[0], out_features=64),
            nn.ReLU(),
            # nn.BatchNorm1d(num_features=64),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            # nn.BatchNorm1d(num_features=32),
            nn.Linear(in_features=32, out_features=action_space.n))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, training: bool):
        logits = self.stack.forward(x)
        if training:
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
        else:
            action = torch.argmax(logits, dim=-1)
        return action


def main():
    env: gym.Env = gym.make("CartPole-v1")
    policy = Policy(observation_space=env.observation_space, action_space=env.action_space)
    experience_store = ExperienceStore()
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    reward_sum_buffer = deque(maxlen=100)

    for update in range(1, 100001):
        experience_store.reset()
        observation = env.reset()
        reward = -1
        done = False
        reward_sum = 0.

        policy.eval()
        while not done:
            action = policy.forward(torch.from_numpy(observation[None, :]), training=True)[0].item()
            experience_store.store(observation, reward, action, done)
            observation, reward, done, info = env.step(action)
            reward_sum += reward

        reward_sum_buffer.append(reward_sum)

        experience_store.finalize(final_reward=reward)

        policy.train()
        logits = policy.stack.forward(experience_store.states)
        logprobs = torch.log_softmax(logits, dim=-1)
        action_mask = F.one_hot(experience_store.actions)
        action_logprobs = torch.sum(logprobs * action_mask, dim=1)
        loss = - torch.mean(action_logprobs * experience_store.returns)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"\r [*] Update {update}/{200}"
              f" - R: {mean(reward_sum_buffer):.0f}"
              f" - L: {loss.item():.4f}", end="")

        if update % 100 == 0:
            print()

        if mean(reward_sum_buffer) >= 200.:
            print("\n [*] Policy converged.")
            return


if __name__ == '__main__':
    main()
