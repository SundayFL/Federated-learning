import inspect
# import start_worker
#
# print(inspect.getsource(start_worker.main))

# Dependencies
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms

use_cuda = torch.cuda.is_available()
torch.manual_seed(1)
device = torch.device("cuda" if use_cuda else "cpu")

import syft as sy
from syft import workers

hook = sy.TorchHook(torch)  # hook torch as always :)

batch_size_test = 1000

optimizer = "SGD"

batch_size = 50
optimizer_args = {"lr": 0.1, "weight_decay": 0.01}
epochs = 1
max_nr_batches = -1  # not used in this example
shuffle = True


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 20, 5, 1)
#         self.conv2 = nn.Conv2d(20, 50, 5, 1)
#         self.fc1 = nn.Linear(4 * 4 * 50, 500)
#         self.fc2 = nn.Linear(500, 10)
#
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = F.relu(self.conv2(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = x.view(-1, 4 * 4 * 50)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)


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

# Instantiate the model
model = Net()

# The data itself doesn't matter as long as the shape is right
mock_data = torch.zeros(batch_size, 1, 28, 28)

# Create a jit version of the model
traced_model = torch.jit.trace(model, mock_data)

type(traced_model)


# Loss function
# @th.jit.script
# def loss_fn(target, pred):
#     out = th.diag(pred[:, target])
#     return -out.sum() / len(out)
#     # return th.nn.functional.nll_loss(pred, target)   #th.tensor(2.5).float()

@torch.jit.script
def loss_fn(pred, target):
    criterion = nn.NLLLoss()
    return criterion(pred, target)
    # return target #torch.tensor(2)   #.nn.functional.nll_loss(input=pred, target=target)

# # @torch.jit.script
# def loss_fn(inputs, outputs):
#     return torch.nn.functional.mse_loss(input=inputs, target=outputs)

type(loss_fn)


train_config = sy.TrainConfig(model=traced_model,
                              loss_fn=loss_fn,
                              optimizer=optimizer,
                              batch_size=batch_size,
                              optimizer_args=optimizer_args,
                              epochs=epochs,
                              shuffle=shuffle)

kwargs_websocket = {"host": "localhost", "hook": hook, "verbose": False}
alice = workers.websocket_client.WebsocketClientWorker(id="alice", port=8777, **kwargs_websocket)

# Send train config
train_config.send(alice)




for epoch in range(10):
    loss = alice.fit(dataset_key="mnist")  # ask alice to train using "xor" dataset
    print("-" * 50)
    print("Iteration %s: alice's loss: %s" % (epoch, loss))

new_model = train_config.model_ptr.get()


def test(model, test_loader):
    model.eval()
    print("Before test")
    print(model)
    print(model.location)
    print(model.owner)
    print(model.training)
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
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

test_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=False, download=True,
                         transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(
                             (0.1307,), (0.3081,))
                         ])), batch_size=batch_size_test, shuffle=True)