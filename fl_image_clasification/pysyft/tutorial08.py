import torch
import torch.nn as nn
import torch.nn.functional as F

import syft as sy  # import the Pysyft library
hook = sy.TorchHook(torch)  # hook PyTorch ie add extra functionalities

# IMPORTANT: Local worker should not be a client worker
hook.local_worker.is_client_worker = False


server = hook.local_worker

x11 = torch.tensor([-1, 2.]).tag('input_data')
x12 = torch.tensor([1, -2.]).tag('input_data2')
# x21 = torch.tensor([-1, 2.]).tag('input_data')
# x22 = torch.tensor([1, -2.]).tag('input_data2')
#
device_1 = sy.VirtualWorker(hook, id="device_1", data=(x11, x12))
# device_2 = sy.VirtualWorker(hook, id="device_2", data=(x21, x22))
# devices = device_1, device_2
#
# @sy.func2plan()
# def plan_double_abs(x):
#     x = x + x
#     x = torch.abs(x)
#     return x
#
# pointer_to_data = device_1.search('input_data')[0]
#
# # Sending non-built Plan will fail
# # try:
# #     plan_double_abs.send(device_1)
# # except RuntimeError as error:
# #     print(error)
#
# plan_double_abs.build(torch.tensor([1., -2.]))
#
# pointer_plan = plan_double_abs.send(device_1)
#
# pointer_to_result = pointer_plan(pointer_to_data)
# print(pointer_to_result)
#
# print(pointer_to_result.get())

class Net(sy.Plan):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 3)
        self.fc2 = nn.Linear(3, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=0)


net = Net()

net.build(torch.tensor([1., 2.]))

# pointer_to_net = net.send(device_1)
#
#
# pointer_to_data = device_1.search('input_data')[0]
# pointer_to_result = pointer_to_net(pointer_to_data)
#
# print(pointer_to_result.get())
# print("end")
