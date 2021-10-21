from statistics import mean

import numpy as np
import torch
import torchvision
from torch import nn
from torch import optim
from torch.utils import data


def categorical_accuracy(pred: torch.Tensor, gt: torch.Tensor):
    if gt.ndim == 2:
        gt = gt.argmax(1)
    eq = (pred.argmax(1) == gt).type(torch.float32)
    return eq.mean()


def preprocess_input(x):
    x = np.array(x).astype("float32")
    x /= 255.
    return torch.Tensor(x)


def get_loader(train: bool):
    dataset = torchvision.datasets.MNIST(root="data", train=train, download=True, transform=preprocess_input)
    loader = data.DataLoader(dataset, batch_size=64, shuffle=True)
    return loader


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=28 * 28, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=10)
        )

    def forward(self, x: torch.Tensor):
        logits = self.stack(x)
        return logits


def train(model: nn.Module,
          train_loader: data.DataLoader,
          val_loader: data.DataLoader,
          loss_fn: nn.Module,
          optimizer: optim.Optimizer,
          train_epochs: int):

    train_steps_per_epoch = len(train_loader)
    val_steps = len(val_loader)

    for epoch in range(1, train_epochs+1):
        print("EPOCH", epoch)

        model.train()
        train_loss = []
        train_acc = []
        for step, (x, y) in enumerate(train_loader, start=1):

            pred: torch.Tensor = model.forward(x)
            loss: torch.Tensor = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = categorical_accuracy(pred, y).item()

            train_loss.append(loss.item())
            train_acc.append(acc)

            if step % 100 == 0:
                print(f"Step {step} / {train_steps_per_epoch} -"
                      f" Loss: {mean(train_loss[-100:]):.4f}"
                      f" Acc: {mean(train_acc[-100:]):>6.2%}")

            if step >= train_steps_per_epoch:
                break

        model.eval()
        test_acc = []
        test_loss = []
        for step, (x, y) in enumerate(val_loader, start=1):
            pred = model.forward(x)
            loss = loss_fn(pred, y).item()
            acc = categorical_accuracy(pred, y).item()
            test_loss.append(loss)
            test_acc.append(acc)
            if step >= val_steps:
                break

        print(f"  Testing - Loss {mean(test_loss):.4f} Acc {mean(test_acc):>6.2%}")
        print()


def main():
    model = Net()
    train_loader = get_loader(train=True)
    val_loader = get_loader(train=False)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=4e-3)
    train(model, train_loader, val_loader, loss_fn, optimizer, train_epochs=30)


if __name__ == '__main__':
    main()
