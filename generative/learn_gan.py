import os
from statistics import mean
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch import nn
from torch import optim
from torch.utils import data


def preprocess_input(x):
    x = np.array(x).astype("float32")
    x /= 255.
    return torch.Tensor(x)


def get_loader(train: bool, batch_size: int):
    dataset = torchvision.datasets.MNIST(root="data", train=train, download=True, transform=preprocess_input)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


class TrainStepReport(NamedTuple):

    generator_loss: float
    critic_loss: float
    critic_acc: float
    critic_real_tpr: float
    critic_fake_tpr: float


class Generator(nn.Module):

    def __init__(self, latent_size: int = 32, learning_rate: float = 3e-4):
        super().__init__()
        self.latent_size = latent_size
        self.stack = nn.Sequential(
            nn.Linear(in_features=latent_size, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=28*28)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, latent_vector: torch.Tensor):
        pixels = self.stack.forward(latent_vector)
        image = pixels.reshape(-1, 28, 28)
        return image

    def generate(self, batch_size: int):
        latent_vector = torch.randn((batch_size, self.latent_size))
        return self.forward(latent_vector)


class Critic(nn.Module):

    def __init__(self, learning_rate: float = 1e-4):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=28*28, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=1))
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, image):
        realness = self.stack.forward(image)
        return realness


class GAN(nn.Module):

    def __init__(self,
                 latent_size: int = 32,
                 generator_lr: float = 1e-4,
                 critic_lr: float = 3e-4,
                 generator_updates_per_step: int = 10,
                 critic_updates_per_step: int = 1):

        super().__init__()
        self.generator = Generator(latent_size, generator_lr)
        self.critic = Critic(critic_lr)
        self.generator_updates_per_step = generator_updates_per_step
        self.critic_updates_per_step = critic_updates_per_step

    def train_step(self, x) -> TrainStepReport:
        batch_size = x.shape[0]

        self.generator.eval()
        fakes = self.generator.generate(batch_size)
        critic_input = torch.cat([x, fakes])
        critic_loss_weights = torch.cat([torch.full((batch_size,), fill_value=-1., dtype=torch.float32),
                                         torch.ones(batch_size, dtype=torch.float32)])
        self.critic.train()
        critic_losses = []
        critic_accs = []
        critic_fake_tpr = []
        critic_real_tpr = []
        for step in range(self.critic_updates_per_step):
            scores = self.critic.forward(critic_input)[..., 0]
            loss = torch.mean(scores * critic_loss_weights)
            self.critic.optimizer.zero_grad()
            loss.backward()
            self.critic.optimizer.step()
            critic_losses.append(loss.item())

            eq = (scores > 0) == (critic_loss_weights < 0)
            critic_accs.append(torch.mean(eq.type(torch.float32)).item())

            classified_real = scores > 0

            real_tp = classified_real[:batch_size]
            fake_tp = torch.logical_not(classified_real[batch_size:])
            critic_real_tpr.append(torch.mean(real_tp.type(torch.float32)).item())
            critic_fake_tpr.append(torch.mean(fake_tp.type(torch.float32)).item())

        self.critic.eval()
        self.generator.train()
        generator_losses = []
        for step in range(self.generator_updates_per_step):
            images = self.generator.generate(batch_size)
            scores = self.critic.forward(images)[..., 0]
            loss = - torch.mean(scores)
            self.generator.optimizer.zero_grad()
            loss.backward()
            self.generator.optimizer.step()
            generator_losses.append(loss.item())

        return TrainStepReport(generator_loss=mean(generator_losses),
                               critic_loss=mean(critic_losses),
                               critic_acc=mean(critic_accs),
                               critic_real_tpr=mean(critic_real_tpr),
                               critic_fake_tpr=mean(critic_fake_tpr))

    def make_album(self, epoch_nr: int):
        os.makedirs("output/gan", exist_ok=True)
        self.eval()
        latent_0 = torch.zeros(self.generator.latent_size // 2)
        latent_1 = torch.zeros(self.generator.latent_size // 2)
        fig, axarr = plt.subplots(5, 5, sharex="all", sharey="all", figsize=(10, 10))
        axiter = iter(axarr.flat)
        for x in range(-2, 3):
            for y in range(-2, 3):
                latent = torch.cat([latent_0 + float(x), latent_1 + float(y)])
                pic = self.generator.forward(latent[None, ...]).detach().numpy()[0]
                ax = next(axiter)
                ax.imshow(pic, cmap="gray", vmin=0, vmax=1)
        plt.suptitle(f"Epoch {epoch_nr}")
        plt.tight_layout()
        plt.savefig(f"output/gan/Epoch_{epoch_nr}_grid.png")
        plt.close()


def main():
    stream = get_loader(train=True, batch_size=32)
    gan = GAN(latent_size=32,
              generator_lr=1e-4,
              critic_lr=1e-4,
              generator_updates_per_step=2,
              critic_updates_per_step=1)
    critic_lr_decay = optim.lr_scheduler.ExponentialLR(gan.critic.optimizer, gamma=0.98)
    generator_lr_decay = optim.lr_scheduler.ExponentialLR(gan.generator.optimizer, gamma=0.98)

    for epoch in range(1, 31):
        gan.make_album(epoch_nr=epoch)
        print(" [*] GAN trainin Epoch", epoch)
        for step, (x, _) in enumerate(stream, start=1):
            report = gan.train_step(x)
            print(f"\rStep {step: >4}/{len(stream)} -"
                  f" Crit acc: {report.critic_acc:>7.2%} -"
                  f" Crit real TPR: {report.critic_real_tpr:>7.2%} -"
                  f" Crit fake TPR: {report.critic_fake_tpr:>7.2%} -"
                  f" Crit loss: {report.critic_loss:.4f} -"
                  f" Gen loss: {report.generator_loss:.4f}",
                  end="")
        print()
        critic_lr_decay.step()
        generator_lr_decay.step()


if __name__ == '__main__':
    main()
