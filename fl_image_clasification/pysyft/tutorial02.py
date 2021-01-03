import torch
from torch import nn
from torch import optim

# A Toy Dataset
data = torch.tensor([[0,0],[0,1],[1,0],[1,1.]])
target = torch.tensor([[0],[0],[1],[1.]])

# A Toy Model
model = nn.Linear(2,1)

def train():
    # Training Logic
    opt = optim.SGD(params=model.parameters(),lr=0.1)
    for iter in range(20):

        # 1) erase previous gradients (if they exist)
        opt.zero_grad()

        # 2) make a prediction
        pred = model(data)

        # 3) calculate how much we missed
        loss = ((pred - target)**2).sum()

        # 4) figure out which weights caused us to miss
        loss.backward()

        # 5) change those weights
        opt.step()

        # 6) print our progress
        print(loss.data)

# train()

# Federated learning way

# import syft as sy
# hook = sy.TorchHook(torch)
#
# # create a couple workers
#
# bob = sy.VirtualWorker(hook, id="bob")
# alice = sy.VirtualWorker(hook, id="alice")
#
# # A Toy Dataset
# data = torch.tensor([[0,0],[0,1],[1,0],[1,1.]], requires_grad=True)
# target = torch.tensor([[0],[0],[1],[1.]], requires_grad=True)
#
# # get pointers to training data on each worker by
# # sending some training data to bob and alice
# data_bob = data[0:2]
# target_bob = target[0:2]
#
# data_alice = data[2:]
# target_alice = target[2:]
#
# # Iniitalize A Toy Model
# model = nn.Linear(2,1)
#
# data_bob = data_bob.send(bob)
# data_alice = data_alice.send(alice)
# target_bob = target_bob.send(bob)
# target_alice = target_alice.send(alice)
#
# # organize pointers into a list
# datasets = [(data_bob,target_bob),(data_alice,target_alice)]
#
# from syft.federated.floptimizer import Optims
# workers = ['bob', 'alice']
# optims = Optims(workers, optim=optim.Adam(params=model.parameters(),lr=0.1))
#
#
# def train():
#     # Training Logic
#     for iter in range(10):
#
#         # NEW) iterate through each worker's dataset
#         for data, target in datasets:
#             # NEW) send model to correct worker
#             model.send(data.location)
#             print(f'Location: {data.location}')
#
#             # Call the optimizer for the worker using get_optim
#             opt = optims.get_optim(data.location.id)
#             # print(data.location.id)
#
#             # 1) erase previous gradients (if they exist)
#             opt.zero_grad()
#
#             # 2) make a prediction
#             pred = model(data)
#
#             # 3) calculate how much we missed
#             loss = ((pred - target) ** 2).sum()
#             print(loss)
#             # 4) figure out which weights caused us to miss
#             loss.backward()
#
#             # 5) change those weights
#             opt.step()
#
#             # NEW) get model (with gradients)
#             model.get()
#
#             # 6) print our progress
#             # print(loss.get().data)  # NEW) slight edit... need to call .get() on loss\
#
# # federated averaging
#
# # train()
#
# Example 3

import torch
import syft as sy
hook = sy.TorchHook(torch)
#
bob = sy.VirtualWorker(hook, id='bob')
alice = sy.VirtualWorker(hook, id='alice')
#
# # this is a local tensor
# x = torch.tensor([1,2,3,4])
# print(x)
#
# # this sends the local tensor to Bob
# x_ptr = x.send(bob)
#
# # this is now a pointer
# print(x_ptr)
#
# # now we can SEND THE POINTER to alice!!!
# pointer_to_x_ptr = x_ptr.send(alice)
#
# print(pointer_to_x_ptr)
#
# print(bob._objects)
# print(alice._objects)

# x is now a pointer to the data which lives on Bob's machine
x = torch.tensor([1,2,3,4,5]).send(bob)

print('  bob:', bob._objects)
print('alice:',alice._objects)

x = x.move(alice)

print('  bob:', bob._objects)
print('alice:',alice._objects)
