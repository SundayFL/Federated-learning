import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import logging
import argparse
import sys
from torch.nn.utils.rnn import pad_sequence

import syft as sy
from syft.workers.websocket_client import WebsocketClientWorker

logger = logging.getLogger(__name__)

LOG_INTERVAL = 25
epochs = 5
use_cuda = False
learning_rate = 0.01
federate_after_n_batches = 1
batch_size = 25


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == "__main__":
    hook = sy.TorchHook(torch)
    kwargs_websocket = {"host": "localhost", "hook": hook}
    alice = WebsocketClientWorker(id="alice", port=8777, **kwargs_websocket)
    device = torch.device("cuda" if use_cuda else "cpu")

    # workers = [alice, bob, charlie]
    workers = [alice]
    grid = sy.PrivateGridNetwork(*workers)

    data = grid.search("#mnist", "#data")
    print(f"Search data: {len(data.keys())}")

    target = grid.search("#mnist", "#target")
    print(f"Search target: {len(target.keys())}")

    datasets_my = []
    for worker in data.keys():
        dataset = sy.BaseDataset(data[worker][0], target[worker][0])
        datasets_my.append(dataset)

    n_features = data['alice'][0].shape[1]
    n_targets = 1

    model = Net()
    if use_cuda:
        model.cuda()

    # Build the FederatedDataset object
    dataset = sy.FederatedDataset(datasets_my)
    print(dataset.workers)
    optimizers = {}
    for worker in dataset.workers:
        optimizers[worker] = optim.Adam(params=model.parameters(), lr=0.02)

    train_loader = sy.FederatedDataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset2 = datasets.MNIST('../data', train=False,
                              transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset2)

    model.train()
    for epoch in range(1, epochs + 1):
        loss_accum = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            model.send(data.location)

            if use_cuda:
                data, target = data.cuda(), target.cuda()

            optimizer = optimizers[data.location.id]
            optimizer.zero_grad()
            pred = model(data)
            loss = F.nll_loss(pred, target)

            loss.backward()
            optimizer.step()

            model.get()
            loss = loss.get()

            loss_accum += float(loss)

            if batch_idx % 8 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tBatch loss: {:.6f}'.format(
                    epoch, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), loss.item()))
            if batch_idx % 80 == 0:
                test(model, device, test_loader)

        print('Total loss', loss_accum)
        test(model, device, test_loader)






