import copy
import logging
import argparse
import re
import sys
import asyncio
import numpy as np
import torch
import random
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from torchvision import datasets, transforms
from pathlib import Path
import json
import os
import codecs
from torchvision.models import vgg11
from model_configurations.simple_cnn import CNN
from model_configurations.mnist_model import MNIST

import syft as sy
from syft.workers import websocket_client
from syft.frameworks.torch.fl import utils


websocket_client.TIMEOUT_INTERVAL = 60
LOG_INTERVAL = 25

logger = logging.getLogger("run_websocket_client")


class LearningMember(object):
    def __init__(self, j):
        self.__dict__ = json.loads(j)

class MockUp():

    def __init__(self, data, targets):
        self.data =data#.astype(np.float32)
        self.targets = targets#.astype('long')

        self.data, self.targets


# Loss function
#@torch.jit.script
#def loss_fn(pred, target):
#    return F.nll_loss(input=pred, target=target.long())


@torch.jit.script
def loss_fn(pred, target):
    return F.cross_entropy(input=pred, target=target.long())

def define_and_get_arguments(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description="Run federated learning using websocket client workers."
    )
    parser.add_argument("--batch_size", type=int, default=32, help="batch size of the training")
    parser.add_argument(
        "--test_batch_size", type=int, default=128, help="batch size used for the test data"
    )
    parser.add_argument(
        "--training_rounds", type=int, default=40, help="number of federated learning rounds"
    )
    parser.add_argument(
        "--federate_after_n_batches",
        type=int,
        default=10,
        help="number of training steps performed on each remote worker before averaging",
    )
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--cuda", action="store_true", help="use cuda")
    parser.add_argument("--seed", type=int, default=1, help="seed used for randomization")
    parser.add_argument("--save_model", action="store_true", help="if set, model will be saved")
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="if set, websocket client workers will be started in verbose mode",
    )

    parser.add_argument("--datapath", help="show program version", action="store", default="../data")
    parser.add_argument(
        "--id", type=str, help="name (id) of the websocket server worker, e.g. --id alice"
    )
    parser.add_argument(
        "--port", "-p", type=int, help="port number of the websocket server worker, e.g. --port 8777"
    )
    parser.add_argument("--epochs", type=int, help="show program version", action="store", default=10)
    parser.add_argument("--public_keys", help="public keys to compute messages", action="store")
    parser.add_argument("--foreign_ids", help="foreign ids to save R values", action="store")
    parser.add_argument("--minimum", help="how many private keys to generate", action="store")
    parser.add_argument("--pathToResources", help="pass path to resources", action="store")

    args = parser.parse_args(args=args)
    return args


async def fit_model_on_worker(
    worker: websocket_client.WebsocketClientWorker,
    traced_model: torch.jit.ScriptModule,
    batch_size: int,
    curr_round: int,
    max_nr_batches: int,
    lr: float,
    epochs: int,
):


    train_config = sy.TrainConfig(
            model=traced_model,
            loss_fn=loss_fn,
            batch_size = batch_size,
            shuffle=True,
            max_nr_batches=max_nr_batches,
            epochs=epochs,
            optimizer="Adam",
            optimizer_args={"lr": lr, "weight_decay": lr*0.1},
    )

    saved_model = copy.deepcopy(traced_model)
    train_config.send(worker)
    loss = await worker.async_fit(dataset_key="mnist", return_ids=[0])
    model = train_config.model_ptr.get().obj
    # to be moved if does not belong here!
    """
        W = np.array([module.weights for module in model.modules() if type(module)!=nn.Sequential])
        publicKeys = // how are they accessed? how do we enter websockets?
        privateValues = [random.random() for x in range(m-1)] // what is m-1?
        R = [sum([privateValues[x]*publicKeys[y]**x for x in range(len(privateValues))])+W for y in range(len(publicKeys))]
    """
    # DP

    # getting old weights
    old_weights = saved_model.state_dict()

    # getting new weights
    new_weights = model.state_dict()

    weights_incr = copy.deepcopy(new_weights)

    # calculating weights increment
    weights_incr = setWeights(old_weights, new_weights, weights_incr, 100)

    # updating weights' increment in returned model
    model.state_dict = weights_incr

    # returning updated weights
    return worker.id, model, loss

# used in differential privacy
def setWeights(list_old, list_new, list_incr, threshold):
    for i, x in enumerate(list_old):
        if np.isscalar(x):
            list_incr[i] = list_new[i] - list_old[i]
            list_incr[i] = list_incr[i] + random.gauss(0, 1)
            if list_incr[i] > threshold:
                list_incr[i] = threshold
            if list_incr[i] < -threshold:
                list_incr[i] = -threshold
        else:
            list_incr[i] = setWeights(list_old[i], list_new[i], list_incr[i], threshold)
    return list_incr

def define_model(model_config, device, modelpath, model_output):
    model_file = Path(modelpath)
    test_tensor = torch.zeros([1, 3, 224, 224])
    if (model_config == 'vgg'):
        model = vgg11(pretrained = True)
        model.classifier[6].out_features = model_output
        print(model.classifier)
        model.eval()
        model.to(device)
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        test_tensor[0] = transform(test_tensor[0])

    if (model_config == 'cnn'):
        model = CNN(3, model_output).to(device)

    if (model_config == 'mnist'):
        model = MNIST().to(device)
        test_tensor = torch.zeros([1, 1, 28, 28])

    if model_file.is_file():
        model.load_state_dict(torch.load(modelpath))
    return model, test_tensor

def define_participant(id, port, **kwargs_websocket):
    worker_instance = sy.workers.websocket_client.WebsocketClientWorker(id=id, port=port, **kwargs_websocket)
    worker_instance.clear_objects_remote()
    return worker_instance

async def main():
    #set up environment
    args = define_and_get_arguments()
    #os.chdir('./akka-server/Client/')
    torch.manual_seed(args.seed)
    print(args)
    hook = sy.TorchHook(torch)
    kwargs_websocket = {"hook": hook, "verbose": args.verbose, "host": 'localhost'}
    #define participants
    worker_instance = define_participant(args.id, args.port, **kwargs_websocket)

    #define model
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model, test_tensor = define_model(args.model_config, device, args.modelpath, int(args.model_output))
    #for p in model.parameters():
    #    p.register_hook(lambda grad: torch.clamp(grad, -6, 6))
    traced_model = torch.jit.trace(model,  test_tensor.to(device))
    traced_model.train()

    # modify the following to get separate models

    learning_rate = args.lr
    worker_id, model, loss_value = fit_model_on_worker(
        worker=worker_instance,
        traced_model=traced_model,
        batch_size=args.batch_size,
        curr_round=curr_round,
        max_nr_batches=args.federate_after_n_batches,
        lr=learning_rate,
        epochs=args.epochs
    )

    # get weights and make R values
    if args.model_config != 'cnn' and args.model_config != 'mnist':
        weights = model.classifier.state_dict()
    else:
        weights = model.fc2.weight.data
    weights = torch.tensor(weights) # how about tensors?
    polynomial = np.array([0 for n in range(public_keys)])
    for m in range(minimum):
        polynomial = np.multiply(polynomial + random.random(), args.public_keys)
    rValues = []
    for n in range(public_keys):
        np.save(pathToResources+id+"/"+id+"_"+foreign_ids[n]+".pt", torch.tensor(weights+polynomial[n]))
    # R values are stored in their own directory in order to simplify storage while working in localhost

if __name__ == "__main__":
    # Logging setup
    FORMAT = "%(asctime)s | %(message)s"
    logging.basicConfig(format=FORMAT)
    logger.setLevel(level=logging.DEBUG)

    # Websockets setup
    websockets_logger = logging.getLogger("websockets")
    websockets_logger.setLevel(logging.INFO)
    websockets_logger.addHandler(logging.StreamHandler())

    # Run main
    asyncio.get_event_loop().run_until_complete(main())