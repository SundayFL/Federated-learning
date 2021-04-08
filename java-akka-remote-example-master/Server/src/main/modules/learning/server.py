import argparse

import torch as th
from syft.workers.websocket_server import WebsocketServerWorker
from torchvision import datasets, transforms
from torch.nn.utils.rnn import pad_sequence

import syft as sy

# Arguments
parser = argparse.ArgumentParser(description="Run websocket server worker.")
parser.add_argument(
    "--port", "-p", type=int, help="port number of the websocket server worker, e.g. --port 8777"
)
parser.add_argument("--host", type=str, default="localhost", help="host for the connection")
parser.add_argument(
    "--id", type=str, help="name (id) of the websocket server worker, e.g. --id alice"
)
parser.add_argument(
    "--verbose",
    "-v",
    action="store_true",
    help="if set, websocket server worker will be started in verbose mode",
)

parser.add_argument("--datapath", help="pass path to data", action="store", default="../data")


def main(datapath, **kwargs):  # pragma: no cover
    """Helper function for spinning up a websocket participant."""

    # Create websocket worker
    worker = WebsocketServerWorker(**kwargs)

    dataset2 = datasets.MNIST(datapath, train=True, transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ))
    data = [x[0] for x in dataset2]
    train_base = sy.BaseDataset(data=data, targets=dataset2.targets)

    # Tell the worker about the dataset
    worker.add_dataset(train_base, key="mnist")

    # Start worker
    worker.start()

    return worker


if __name__ == "__main__":
    hook = sy.TorchHook(th)

    args = parser.parse_args()
    kwargs = {
        "id": args.id,
        "host": args.host,
        "port": args.port,
        "hook": hook,
        "verbose": args.verbose,
    }

    main(args.datapath, **kwargs)
