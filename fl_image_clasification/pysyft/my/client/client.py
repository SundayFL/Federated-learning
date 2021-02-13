print("Start hack")
def script_method(fn, _rcb=None):
    return fn
def script(obj, optimize=True, _frames_up=0, _rcb=None):
    return obj
import torch.jit
torch.jit.script_method = script_method
torch.jit.script = script
print("End hack")

import shaloop

# import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import logging

import syft as sy
# from syft.workers.websocket_client import WebsocketClientWorker
# sy.workers.websocket_client
import sys
from argparse import ArgumentParser


logger = logging.getLogger(__name__)

LOG_INTERVAL = 25
epochs = 40
use_cuda = False
learning_rate = 0.02
federate_after_n_batches = 50
batch_size = 25

class Net(nn.Module):
    """
    Simple Triple-layered-FCN from MNIST (784) to 10 class.
    """
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, 3, 1)
#         self.conv2 = nn.Conv2d(32, 64, 3, 1)
#         self.dropout1 = nn.Dropout(0.25)
#         self.dropout2 = nn.Dropout(0.5)
#         self.fc1 = nn.Linear(9216, 128)
#         self.fc2 = nn.Linear(128, 10)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = self.conv2(x)
#         x = F.relu(x)
#         x = F.max_pool2d(x, 2)
#         x = self.dropout1(x)
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.dropout2(x)
#         x = self.fc2(x)
#         output = F.log_softmax(x, dim=1)
#         return output

def epoch_total_size(data):
    total = 0
    for i in range(len(data)):
        for j in range(len(data[i])):
            total += data[i][j].shape[0]

    return total


def train(epoch, model, data, target, optimizer, criterion):
    model.train()

    nr_batches = federate_after_n_batches
    epoch_total = epoch_total_size(data)
    current_epoch_size = 0

    batch = [torch.Tensor(t)for t in data[0][0:batch_size]]
    print("ok")

    batch = torch.nn.utils.rnn.pad_sequence(batch)
    print(f"batch {batch}")

    for i in range(len(data)):
        for j in range(len(data[i])):
            current_epoch_size += len(data[i][j])
            worker = data[i][j].location

            model.send(worker)
            optimizer.zero_grad()

            if use_cuda:
                data, target = data[i][j].cuda(), target[i][j].cuda()

            pred = model(data[i][j])
            # pred = model(batch)
            print("pred")
            loss = F.nll_loss(pred, target[i][j])
            loss.backward()
            optimizer.step()

            model.get()
            loss = loss.get()
            print(f'{i} Train Epoch: {epoch} | With {worker.id} data |: [{current_epoch_size}/{epoch_total} '
                  f'({100. * current_epoch_size / epoch_total}%)]\tLoss: {loss.item()}')

            if j == 1000:
                return model

            # if (j + 1) %100 == 0:
            #     # print(f"Get on {j} {worker}")
            #     model.get()
            #     loss = loss.get()
            #     print(f'{i} Train Epoch: {epoch} | With {worker.id} data |: [{current_epoch_size}/{epoch_total} '
            #           f'({100. * current_epoch_size / epoch_total}%)]\tLoss: {loss.item()}')
            #
            # if j == 10000:
            #     model.get()
            #     return model

    return model


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item() # sum up batch loss
            pred = output.argmax(1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            test_loss, correct, len(test_loader.dataset), accuracy
        )
    )


def main(data_path):
    hook = sy.TorchHook(torch)
    kwargs_websocket = {"host": "localhost", "hook": hook}
    # kwargs_websocket = {"host": "localhost"}
    alice = sy.workers.websocket_client.WebsocketClientWorker(id="alice", port=8777, **kwargs_websocket)
    device = torch.device("cuda" if use_cuda else "cpu")

    # workers = [alice, bob, charlie]
    workers = [alice]
    grid = sy.PrivateGridNetwork(*workers)

    data = grid.search("#mnist", "#data")
    print(f"Search data: {len(data.keys())}")

    target = grid.search("#mnist", "#target")
    print(f"Search target: {len(target.keys())}")

    model = Net().to(device)

    data = list(data.values())
    target = list(target.values())
    epoch_total = epoch_total_size(data)
    print(f"Total epochs: {epoch_total}")

    if use_cuda:
        model.cuda()
    optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            data_path,
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=1,
        shuffle=True,
    )

    for epoch in range(1, epochs + 1):
        m = train(epoch, model, data, target, optimizer, criterion)
        test(m, device, test_loader)


if __name__ == "__main__":
    print("Start");
    print('Number of arguments: ' + str(len(sys.argv)) + ' arguments.')
    print('Argument List:' + str(sys.argv))

    parser = ArgumentParser()
    parser.add_argument("--datapath", help="show program version", action="store", default="../data")

    args = parser.parse_args()

    # Check for --version or -V
    if args.datapath:
        print(args.datapath)
    else:
        print("no arg")

    FORMAT = "%(asctime)s %(levelname)s %(filename)s(l:%(lineno)d) - %(message)s"
    LOG_LEVEL = logging.DEBUG
    logging.basicConfig(format=FORMAT, level=LOG_LEVEL)

    websockets_logger = logging.getLogger("websockets")
    websockets_logger.setLevel(logging.DEBUG)
    websockets_logger.addHandler(logging.StreamHandler())

    main(args.datapath)
