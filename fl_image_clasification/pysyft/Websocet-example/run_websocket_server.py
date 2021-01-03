from multiprocessing import Process
import syft as sy
from syft.workers.websocket_server import WebsocketServerWorker
import torch
import argparse
import os
from torchvision import datasets, transforms
from syft.generic.pointers.callable_pointer import create_callable_pointer
hook = sy.TorchHook(torch)


def start_proc(participant, kwargs):  # pragma: no cover
    """ helper function for spinning up a websocket participant """

    def target():
        server = participant(**kwargs)

        # create_callable_pointer(server, "alice", "alice", server, "#mnist", "The images in the MNIST training dataset.")

        test_loader = torch.utils.data.DataLoader(datasets.MNIST('../data'))
        server.load_data([test_loader.dataset])
        print(f"Type of mnist dataset: {type(test_loader.dataset)}")
        #
        # x = torch.tensor([1, 2, 3, 4, 5]).tag("#fun", "#mnist").describe("The images in the MNIST training dataset.")
        # y = torch.tensor([5, 4, 3, 2, 1]).tag("#fun", "#mnist").describe("The images in the MNIST training dataset.")
        #
        # print(f"Type of x: {type(x)}")
        # # x.send(server)
        #
        # server.load_data([x, y])

        server.start()

    p = Process(target=target)
    p.start()
    return p


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

args = parser.parse_args()

kwargs = {
    "id": args.id,
    "host": args.host,
    "port": args.port,
    "verbose": args.verbose,
}

kwargs = {
    "id": "worker",
    "host": "localhost",
    "port": "8777",
    "hook": hook
}

if os.name != "nt":
    print("not nt")
    server = start_proc(WebsocketServerWorker, kwargs)
else:
    print("nt")
    server = WebsocketServerWorker(**kwargs)
    server.start()