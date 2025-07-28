#!/usr/bin/env python3

import torch
import torch.nn as nn

device = torch.device('mps')

class CNN(nn.Module):

    def __init__(self, kernel_size):
        """Initilize the Network"""
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, kernel_size)
        self.conv2 = nn.Conv2d(4, 5, kernel_size)
        self.conv3 = nn.Conv2d(5, 5, kernel_size)
        self.lin_act = nn.ReLU()
        self.avPool = nn.AveragePool2d(25)
        self.pool = nn.MaxPool2d(25)
        self.probs = nn.Sigmoid()

    def forward(self, x):
        """Forward activation of network"""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.lin_act(x)
        x = self.avPool(x)
        x = self.pool(x)
        x = self.provs(x)
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

             optim.zero_grad()

             outputs = model(inputs)
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

        print(f'Epoch {epoch + 1} completed. Train Loss: {epoch_loss:.3f}, Train Accuracy: {epoch_acc:.2f}%')
    return train_losses, train_accs
