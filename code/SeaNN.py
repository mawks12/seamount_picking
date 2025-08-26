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
        self.conv1 = nn.Conv2d(1, 3, kernel_size, padding=4)
        self.conv1Act = nn.ReLU()
        self.conv2 = nn.Conv2d(3, 5, kernel_size, padding=4)
        self.conv2Act = nn.ReLU()
        self.conv3 = nn.Conv2d(5, 3, kernel_size, padding=4)
        self.conv3Act = nn.ReLU()
        self.flat = nn.Conv2d(3, 1, kernel_size, padding=4)
        self.probs = nn.Sigmoid()

    def forward(self, x):
        """Forward activation of network"""
        x = self.conv1(x)
        x = self.conv1Act(x)
        x = self.conv2(x)
        x = self.conv2Act(x)
        x = self.conv3(x)
        x = self.conv3Act(x)
        x = self.flat(x)
        #x = self.reduce(x)
        x = self.probs(x)
        return x


def train(model, trainloader, num_epoch, device):
    """Train a network"""

    loss_func = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters())

    model.train()
    model.to(device)

    train_losses = []
    train_accuracies = []

    for epoch in range(num_epoch):
        running_loss = 0
        correct = 0
        total = 0

        for i, data in enumerate(trainloader):
             inputs, labels = data[0].to(device), data[1].to(device)

             optim.zero_grad()

             outputs = model(inputs)
             loss = loss_func(outputs, labels)

             loss.backward()
             optim.step()

             running_loss += loss.item()
             predicted = torch.where(outputs > 0.5, 1, 0)
             total += 1
             correct += (predicted == labels).sum().item() / labels.shape[0]

             if i % 100 == 99:
                batch_loss = running_loss / 100
                batch_acc = 100 * correct / total
                print(f'Epoch: {epoch + 1}, Batch: {i + 1}, Loss: {batch_loss:.3f}, Accuracy: {batch_acc:.2f}%')
                running_loss = 0.0
                correct = 0
                total = 0

        epoch_loss, epoch_acc = running_loss / total, correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        print(f'Epoch {epoch + 1} of {num_epoch} completed. Train Loss: {epoch_loss:.3f}, Train Accuracy: {epoch_acc:.2f}%')
    return train_losses, train_accuracies


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
        data = torch.reshape(data, (1, data.shape[0], data.shape[1]))
        labels = torch.reshape(labels, (1, labels.shape[0], labels.shape[1])).type(torch.float32)
        # if self.transform:
        #     data = self.transform(data)
        return data, labels
