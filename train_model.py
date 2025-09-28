import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
from torch.utils.data import Dataset, DataLoader


'''
Alex Knowlton
12/6/24

This module defines a PyTorch neural network, imports the generated
communications dataset, and evaluates the network performance on the data.

Network Structure:
Convolutional layer:
    Input: 512 points, with 2 channels
    Input size: 512x2
    Kernel: size (1, 2), so operate on one point at a time
    # channels: 48, representing a bunch of different points in constellation
    No padding,
    Standard stride of 1
'''


class CommsDataset(Dataset):
    '''
    Defines a dataset that takes in the communications dataset as numpy arrays,
    not tensors.
    '''

    def __init__(self, features, labels):
        '''
        Class initializer
        Inputs:
            features: NxD numpy features array
            labels: (N,) numpy labels array with class numbers
        '''
        self.features = torch.tensor(features, dtype=torch.float)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.height = 512
        self.width = 2

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        constellation = self.features[index].reshape(1, 2, 512)
        label = self.labels[index]
        return constellation, label


class CommsNet(nn.Module):
    '''
    Neural network for constellation prediction
    '''

    def __init__(self):
        super(CommsNet, self).__init__()
        self.n_channels = 64
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.n_channels,
                      kernel_size=(2, 1), stride=1, padding=0)
        )

        self.fc = nn.Sequential(
            nn.Linear(self.n_channels, 48),
            nn.ReLU(),

            nn.Linear(48, 48),
            nn.ReLU(),

            nn.Linear(48, 6)
        )

    def forward(self, x):
        '''
        Forward pass of X vector
        '''
        conv_output = self.conv(x)
        softmax_output = F.softmax(conv_output, dim=1)
        x = torch.sum(softmax_output * conv_output, dim=3, keepdim=True)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def train(trainloader, net, criterion, optimizer, device):
    start = time.time()
    for epoch in range(10):  # loop over the dataset multiple times
        start = time.time()
        running_loss = 0.0
        for i, (images, labels) in enumerate(trainloader):

            # send images and labels to device
            images = images.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward pass
            yhat = net.forward(images)

            # loss function - cross entropy
            loss = criterion(yhat, labels)

            # backward pass
            loss.backward()

            # optimize the network
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            if i % 100 == 99:    # print every 2000 mini-batches
                end = time.time()
                print('[epoch %d, iter %5d] loss: %.5f, time: %.2f seconds' %
                      (epoch + 1, i + 1, running_loss / 100, end-start))
                start = time.time()
                running_loss = 0.0
    stop = time.time()
    print(f'Finished Training: {(stop - start) / 60} minutes')


def test(testloader, net, device, is_test=True):
    data_name = 'test' if is_test else 'training'
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy: {data_name} constellations: %.2f %%' % (
        100 * correct / total))
    return correct / total


def train_model():
    X_train = np.load('./data/train_data.npy')
    y_train = np.load('./data/train_results.npy')

    trainset = CommsDataset(X_train, y_train)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = CommsNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    train(trainloader, net, criterion, optimizer, device)
    return net


def test_model(net):
    # X_train = np.load('./data/train_data.npy')
    X_test = np.load('./data/test_data.npy')
    # y_train = np.load('./data/train_results.npy')
    y_test = np.load('./data/test_results.npy')

    # trainset = CommsDataset(X_train, y_train)
    testset = CommsDataset(X_test, y_test)
    # trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=32, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # net = CommsNet().to(device)
    # criterion = nn.CrossEntropyLoss()

    # optimizer = optim.Adam(net.parameters(), lr=0.001)
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # # train(trainloader, net, criterion, optimizer, device)
    # train_error = test(trainloader, net, device, is_test=False)
    test_error = test(testloader, net, device, is_test=True)
    # return train_error, test_error
    return test_error


if __name__ == "__main__":
    test_model()
