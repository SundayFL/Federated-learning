import torch as th
import syft as sy
sy.create_sandbox(globals(), verbose=False)
epochs = 50

hook = sy.TorchHook(th)  # <-- NEW: hook PyTorch ie add extra functionalities to support Federated Learning
bob = sy.VirtualWorker(hook, id="bob")  # <-- NEW: define remote worker bob
alice = sy.VirtualWorker(hook, id="alice")  # <-- NEW: and alice

grid = sy.PrivateGridNetwork(bob, alice)

boston_data = grid.search("#boston", "#data")
boston_target = grid.search("#boston", "#target")

n_features = boston_data['alice'][0].shape[1]
n_targets = 1

model = th.nn.Linear(n_features, n_targets)


# Cast the result in BaseDatasets
datasets = []
for worker in boston_data.keys():
    print(boston_data[worker][0])
    dataset = sy.BaseDataset(boston_data[worker][0], boston_target[worker][0])
    datasets.append(dataset)

# Build the FederatedDataset object
dataset = sy.FederatedDataset(datasets)
print(dataset.workers)
optimizers = {}
for worker in dataset.workers:
    optimizers[worker] = th.optim.Adam(params=model.parameters(),lr=1e-2)


train_loader = sy.FederatedDataLoader(dataset, batch_size=32, shuffle=False, drop_last=False)

for epoch in range(1, epochs + 1):
    loss_accum = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        print(len(data))
        model.send(data.location)

        optimizer = optimizers[data.location.id]
        optimizer.zero_grad()
        pred = model(data)
        loss = ((pred.view(-1) - target) ** 2).mean()
        loss.backward()
        optimizer.step()

        model.get()
        loss = loss.get()

        loss_accum += float(loss)

        if batch_idx % 8 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tBatch loss: {:.6f}'.format(
                epoch, batch_idx, len(train_loader),
                100. * batch_idx / len(train_loader), loss.item()))

    print('Total loss', loss_accum)