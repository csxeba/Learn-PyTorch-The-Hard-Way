import os
from statistics import mean

import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils import data
from torch import nn, optim


def preprocess_input(x):
    x = np.array(x).astype("float32")
    x /= 255.
    return torch.Tensor(x)


def get_loader(train: bool, batch_size: int):
    dataset = torchvision.datasets.MNIST(root="data", train=train, download=True, transform=preprocess_input)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


def sum_of_square_errors(pred, gt):
    dif = (pred - gt) ** 2.
    sse = torch.sum(dif, dim=(1, 2))
    return sse


def kl_divergence(z_mean: torch.Tensor, z_log_var: torch.Tensor):
    z = 1. + z_log_var - z_mean ** 2. - torch.exp(z_log_var)
    kl = -0.5 * torch.sum(z, dim=1)
    return kl


class VAE(nn.Module):

    def __init__(self, latent_size: int = 32):
        super().__init__()

        self.latent_size = latent_size

        self.encoder_backbone = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=28*28, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
        )
        self.vae_mean = nn.Linear(in_features=32, out_features=latent_size)
        self.vae_log_var = nn.Linear(in_features=32, out_features=latent_size)

        self.decoder_dense = nn.Sequential(
            nn.Linear(in_features=latent_size, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=28*28),
            nn.Sigmoid()
        )

    def train_forward(self, x: torch.Tensor):
        x_features = self.encoder_backbone.forward(x)

        x_mean = self.vae_mean.forward(x_features)
        x_log_var = self.vae_log_var.forward(x_features)
        x_std = torch.sqrt(torch.exp(x_log_var))

        x = torch.randn(size=x_mean.shape, dtype=x_mean.dtype) * x_std + x_mean

        reconstruction_features = self.decoder_dense.forward(x)
        reconstruction = reconstruction_features.reshape((-1, 28, 28))

        return reconstruction, x_mean, x_log_var

    def reconstruct(self, x):
        x_features = self.encoder_backbone.forward(x)
        x_mean = self.vae_mean.forward(x_features)
        reconstruction_features = self.decoder_dense.forward(x_mean)
        reconstruction = reconstruction_features.reshape((-1, 28, 28))
        return reconstruction

    def generate(self, latent_vector=None):
        if latent_vector is None:
            latent_vector = torch.randn(size=(1, self.latent_size))
        reconstruction_features = self.decoder_dense.forward(latent_vector)
        reconstruction = reconstruction_features.reshape((-1, 28, 28))
        return reconstruction

    def generate_grid(self):
        latent = torch.full((1, self.latent_size), 0.0, dtype=torch.float32)
        rows = []
        for x in torch.arange(-2.0, 2.0, step=0.4):
            row = []
            for y in torch.arange(-2.0, 2.0, step=0.4):
                latent[0, 0] += x
                latent[0, 1] += y
                image = self.generate(latent)
                row.append(image[0])
                latent[0, 0:2] = 0.
            rows.append(torch.cat(row, dim=0))
        grid = torch.cat(rows, dim=1).detach().numpy()
        grid = np.clip(grid, 0, 1) * 255
        return grid.astype("uint8")


def main():

    BATCH_SIZE = 32
    LATENT_SIZE = 2
    BETA = 1.

    data = get_loader(train=True, batch_size=BATCH_SIZE)
    model = VAE(latent_size=LATENT_SIZE)
    optimizer = optim.Adam(params=model.parameters(), lr=1e-3)
    grids = []
    for epoch in range(1, 31):
        print("Epoch", epoch, "/", 30)
        mle_losses = []
        kld_losses = []
        model.train()
        for step, (x, _) in enumerate(data, start=1):

            if len(x) < BATCH_SIZE:
                continue

            reconstruction, dist_mean, dist_log_var = model.train_forward(x)
            reconstruction_loss = sum_of_square_errors(reconstruction, x)
            kl_div = kl_divergence(dist_mean, dist_log_var)
            total_loss = torch.mean(reconstruction_loss + kl_div * BETA)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            mle_losses.append(reconstruction_loss.detach().numpy().mean())
            kld_losses.append(kl_div.detach().numpy().mean())

            print(f"\r Step {step}/{len(data)} -"
                  f" MLE: {mean(mle_losses[-100:]):.4f}"
                  f" KLD: {mean(kld_losses[-100:]):.4f}", end="")

            if step % 100 == 0:
                print()

        print()
        model.eval()
        grids.append(model.generate_grid())
        os.makedirs("output/vae", exist_ok=True)
        Image.fromarray(grids[-1]).save(f"output/vae/epoch_{epoch}.png")


if __name__ == '__main__':
    main()
