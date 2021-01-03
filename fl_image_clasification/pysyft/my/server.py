from multiprocessing import Process
import syft as sy
from syft.workers.websocket_server import WebsocketServerWorker
import torch
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
hook = sy.TorchHook(torch)


def start_proc(participant, kwargs):  # pragma: no cover
    """ helper function for spinning up a websocket participant """

    def target():
        server = participant(**kwargs)

        data = [x[0] for x in datasets.MNIST('../data', train=True, transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ))]
        data1 = pad_sequence(data, batch_first=True).tag('#mnist', '#data')

        target_data = [torch.LongTensor([x[1]]).tag('#mnist', '#data') for x in datasets.MNIST('../data', train=True, transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))]
        target_data1 = torch.cat(target_data).tag('#mnist', '#target')
        # print(target_data)
        # server.load_data(data[:1000])
        # server.load_data(target_data[:1000])
        server.load_data([data1])
        server.load_data([target_data1])

        print("Starting")

        server.start()

    p = Process(target=target)
    p.start()
    return p


kwargs = {
    "id": "alice",
    "host": "localhost",
    "port": "8777",
    "hook": hook
}

server1= start_proc(WebsocketServerWorker, kwargs)
