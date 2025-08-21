#!/usr/bin/env python3

import os
from pathlib import Path
import xarray as xr
import numpy as np
import torch
from torch.utils.data import Sampler
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.transforms import v2

device = torch.device('mps')

class CNN(nn.Module):

    def __init__(self, kernel_size):
        """Initilize the Network"""
        super().__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size)
        self.conv2 = nn.Conv2d(3, 5, kernel_size)
        self.conv3 = nn.Conv2d(5, 3, kernel_size)
        self.lin_act = nn.ReLU()
        self.pool = nn.MaxPool1d(5)
        self.flat = nn.Flatten()
        self.probs = nn.Sigmoid()

    def forward(self, x):
        """Forward activation of network"""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.lin_act(x)
        # x = self.pool(x)
        x = self.reduce(x)
        x = self.probs(x)
        return x


def train(model, trainloader, num_epoch, device):
    """Train a network"""

    loss_func = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters())

    model.train()

    train_losses = []
    train_accuracies = []

    for epoch in range(num_epoch):
        running_loss = 0.
        correct = 0
        total = 0

        for i, data in enumerate(trainloader):
             inputs, labels = data[0].to(device), data[1].to(device)

             print(inputs.shape)
             optim.zero_grad()

             outputs = model(inputs)
             print(outputs.shape, labels.shape)
             loss = loss_func(outputs, labels)

             loss.backward()
             optim.step()

             running_loss += loss.item()
             _, predicted = torch.where(outputs > 0.5, 1, 0)
             total += labels.size(0)
             correct += (predicted == labels).sum().item()

             if i % 100 == 99:
                batch_loss = running_loss / 100
                batch_acc = 100 * correct / total
                print(f'Epoch: {epoch + 1}, Batch: {i + 1}, Loss: {batch_loss:.3f}, Accuracy: {batch_acc:.2f}%')
                running_loss = 0.0
                correct = 0
                total = 0

        epoch_loss, epoch_acc = running_loss / total, 100 * correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        print(f'Epoch {epoch + 1} of {num_epoch} completed. Train Loss: {epoch_loss:.3f}, Train Accuracy: {epoch_acc:.2f}%')
    return train_losses, train_accs


def evaluate(model, dataloader, device):
    """Evaluate the model"""

    criterion = nn.CrossEntropyLoss()

    model.eval()
    runnimg_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data in dataloader:
            points, labels = data[0].to(device), data[1].to(device)

            outputs = model(points)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predited = torch.where(outputs > 0.5, 1, 0)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

class SeamountDataset(Dataset):
    """Seamount Dataset for pytorch"""

    def __init__(self, data_dir, vgg_file, transform=v2.RandomCrop(size=20)):
        self.data_dir = Path(data_dir)
        self.samples = sorted(os.listdir(data_dir))
        self.vgg_file = vgg_file
        self.transform = transform
        super().__init__()
        return

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = xr.open_dataset(self.data_dir / self.samples[idx])
        data = torch.as_tensor(item.z.values)
        labels = torch.as_tensor(item.Labels.values)
        assert data.shape == labels.shape
        # if self.transform:
        #     data = self.transform(data)
        return data, labels
