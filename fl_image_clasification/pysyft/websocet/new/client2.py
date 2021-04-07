
import torch
import syft as sy
from syft import workers
from torch import nn
import asyncio
from syft.frameworks.torch.fl import utils

hook = sy.TorchHook(torch)


kwargs_websocket = {"host": "localhost", "hook": hook, "verbose": False}
alice = workers.websocket_client.WebsocketClientWorker(id="alice", port=8777, **kwargs_websocket)


# Model and Loss Function
def loss_fn(pred, target):
    return torch.nn.functional.nll_loss(input=pred, target=target)


class Net(nn.Module):
    def __init__(self, D_in, D_out):
        super(Net, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, D_out)
        self.activation = torch.nn.Sigmoid()
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        return x


#Training Configuration
async def fit_model_on_worker(
    worker: workers.websocket_client.WebsocketClientWorker,
    traced_model: torch.jit.ScriptModule,
    batch_size: int,
    curr_round: int,
    max_nr_batches: int,
    lr: float,
):
    train_config = sy.TrainConfig(
        model=traced_model,
        loss_fn=loss_fn,
        batch_size=batch_size,
        shuffle=True,
        max_nr_batches=max_nr_batches,
        epochs=1,
        optimizer="SGD",
        optimizer_args={"lr": lr},
    )
    train_config.send(worker)
    loss = await worker.async_fit(dataset_key="LoanRiskDataset", return_ids=[0])
    model = train_config.model_ptr.get().obj
    return worker.id, model, loss


results = await asyncio.gather(
        *[
            fit_model_on_worker(
                worker=worker,
                traced_model=traced_model,
                batch_size=args.batch_size,
                curr_round=curr_round,
                max_nr_batches=args.federate_after_n_batches,
                lr=learning_rate,
            )
            for worker in worker_instances
        ]
    )

models = {}
loss_values = {}

for worker_id, worker_model, worker_loss in results:
        if worker_model is not None:
            models[worker_id] = worker_model
            loss_values[worker_id] = worker_loss
traced_model = utils.federated_avg(models)