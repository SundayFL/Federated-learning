print("Start hack")
def script_method(fn, _rcb=None):
    return fn
def script(obj, optimize=True, _frames_up=0, _rcb=None):
    return obj
import torch.jit
torch.jit.script_method = script_method
torch.jit.script = script
print("End hack")

from multiprocessing import Process
from syft import TorchHook
# import syft as sy
from syft.workers.websocket_server import WebsocketServerWorker
from torchvision import datasets, transforms
from torch.nn.utils.rnn import pad_sequence
from argparse import ArgumentParser

hook = TorchHook(torch)

def start_proc(participant, kwargs, datapath):  # pragma: no cover
    """ helper function for spinning up a websocket participant """

    def target():
        server = participant(**kwargs)

        data = [x[0] for x in datasets.MNIST(datapath, train=True, transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ))]
        data1 = pad_sequence(data, batch_first=True).tag('#mnist', '#data')

        target_data = [torch.LongTensor([x[1]]).tag('#mnist', '#data') for x in datasets.MNIST(datapath, train=True, transform=transforms.Compose(
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


parser = ArgumentParser()
parser.add_argument("--datapath", help="pass path to data", action="store", default="../data")
parser.add_argument("--id", help="id", action="store", default="alice")
parser.add_argument("--host", help="host", action="store", default="localhost")
parser.add_argument("--port", help="port", action="store", default="8777")

args = parser.parse_args()

# Check for --version or -V
print(f"datapath: {args.datapath}, id: {args.id}, host: {args.host}, port: {args.port}")

kwargs = {
    "id": args.id,
    "host": args.host,
    "port": args.port,
    "hook": hook
}

server1= start_proc(WebsocketServerWorker, kwargs, args.datapath)
