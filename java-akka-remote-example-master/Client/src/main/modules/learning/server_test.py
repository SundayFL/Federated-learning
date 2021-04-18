import argparse

import torch as th
from syft.workers.websocket_server import WebsocketServerWorker
from torchvision import datasets, transforms
from torch.nn.utils.rnn import pad_sequence

import syft as sy

import torch
import tensorflow_federated as tff

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
parser.add_argument("--data_set_id", type=int, help="id of data set", action="store", default=0)


def main(datapath, data_set_id, **kwargs):  # pragma: no cover
    """Helper function for spinning up a websocket participant."""

    # Create websocket worker
    worker = WebsocketServerWorker(**kwargs)

    print(data_set_id)
    emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()
    example_dataset = emnist_train.create_tf_dataset_for_client(emnist_train.client_ids[data_set_id])

    data = [torch.FloatTensor(x['pixels'].numpy()).unsqueeze(0) for x in example_dataset]
    targets = [torch.tensor(x['label'].numpy(), dtype=torch.long) for x in example_dataset]

    train_base = sy.BaseDataset(data=data, targets=targets)

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
        "verbose": args.verbose
    }

    main(args.datapath, args.data_set_id, **kwargs)
