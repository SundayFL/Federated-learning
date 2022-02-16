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
from model_configurations.mimic_model import MIMIC
from model_configurations.chess_model import Chess
from model_configurations.nnd_model import Neural_Network_deep

import syft as sy
from syft.workers import websocket_client
from syft.frameworks.torch.fl import utils


websocket_client.TIMEOUT_INTERVAL = 600
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
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--cuda", action="store_true", help="use cuda")
    parser.add_argument("--seed", type=int, default=1, help="seed used for randomization")
    parser.add_argument("--save_model", action="store_false", help="if set, model will be saved")
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
    parser.add_argument("--model_config", default="vgg")
    parser.add_argument("--model_output", default="12")
    parser.add_argument("--pathToResources", help="pass path to resources", action="store")
    parser.add_argument("--diff_priv", help="whether to include differential privacy", action="store")
    parser.add_argument("--dp_noise_std", help="standard deviation for differential privacy noise", action="store")
    parser.add_argument("--dp_threshold", help="threshold for differential privacy max weight incr", action="store")
    parser.add_argument("--learningTaskId", default="mnist")


    args = parser.parse_args(args=args)
    return args


async def fit_model_on_worker(
    worker: websocket_client.WebsocketClientWorker,
    traced_model: torch.jit.ScriptModule,
    batch_size: int,
    max_nr_batches: int,
    lr: float,
    epochs: int,
    diff_priv: bool,
    dp_noise_std: float,
    dp_threshold: float,
    learningTaskId: str
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

    train_config.send(worker)

    print("STEP C - before fit")
    loss = await worker.async_fit(dataset_key=learningTaskId, return_ids=[0])
    print("STEP D - after fit")
    model = train_config.model_ptr.get().obj


    # Differential Privacy
    if diff_priv=="True":
        print("Differential privacy in use")
        # getting old weights
        old_weights = traced_model.state_dict()

        # getting new weights
        new_weights = model.state_dict()

        weights_incr = copy.deepcopy(new_weights)

        # calculating weights increment
        for layer in weights_incr:
            weights_incr[layer] = weights_incr[layer] + torch.tensor(setWeights(
                np.array(old_weights[layer]),
                np.array(new_weights[layer]),
                np.array(weights_incr[layer]),
                dp_noise_std,
                dp_threshold
            ))

        # updating weights' increment in returned model
        model.load_state_dict(weights_incr)
    else:
        print("Differential privacy out of use")


    # returning updated weights
    return worker.id, model, loss

# DIfferential Privacy implementation
def setWeights(list_old, list_new, list_incr, std, threshold):
    eps = 0.01  # to enable switching signs by weights
    for i, x in enumerate(list_old):
        if np.isscalar(x):
            list_incr[i] = list_new[i] - list_old[i]
            list_incr[i] = list_incr[i] + random.gauss(0, std) # adding noise

            # setting threshold
            if list_old[i]*threshold < 0:
                smaller_boundary = list_old[i]*threshold - eps
                higher_boundary = - list_old[i]*threshold + eps
            else:
                smaller_boundary = -list_old[i]*threshold - eps
                higher_boundary = list_old[i]*threshold + eps

            if list_incr[i] > higher_boundary:
                list_incr[i] = higher_boundary
            elif list_incr[i] < smaller_boundary:
                list_incr[i] = smaller_boundary
        else:
            list_incr[i] = setWeights(list_old[i], list_new[i], list_incr[i], std, threshold)
    return list_incr

def define_model(model_config, device, model_output):
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

    if (model_config == 'chess'):
        model = Chess().to(device)
        test_tensor = torch.zeros([1, 1, 64, 64])

    if (model_config == 'mimic'):
        model = MIMIC().to(device)
        test_tensor = torch.zeros([1, 48, 19])
    if (model_config == 'nnd'):
        model = Neural_Network_deep().to(device)
        test_tensor = torch.zeros([1, 1, 28, 28])
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
    if not os.path.exists(args.pathToResources+args.id):
        os.mkdir(args.pathToResources+args.id)
    hook = sy.TorchHook(torch)
    kwargs_websocket = {"hook": hook, "verbose": args.verbose, "host": 'localhost'}
    #define participants
    worker_instance = define_participant(args.id, args.port, **kwargs_websocket)

    #define model
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model, test_tensor = define_model(args.model_config, device, int(args.model_output))
    #for p in model.parameters():
    #    p.register_hook(lambda grad: torch.clamp(grad, -6, 6))
    traced_model = torch.jit.trace(model,  test_tensor.to(device))
    traced_model.train()

    learning_rate = args.lr
    worker_id, model, loss_value = await fit_model_on_worker(
        worker=worker_instance,
        traced_model=traced_model,
        batch_size=args.batch_size,
        max_nr_batches=args.federate_after_n_batches,
        lr=learning_rate,
        epochs=args.epochs,
        diff_priv=args.diff_priv,
        dp_noise_std=float(args.dp_noise_std),
        dp_threshold=float(args.dp_threshold),
        learningTaskId=args.learningTaskId
    )

    # get weights and make R values
    """if args.model_config != 'cnn' and args.model_config != 'mnist':
        weights = model.classifier.state_dict()
    else:
        weights = model.fc2.weight.data"""

    weights = model.state_dict()
    torch.save(weights, args.pathToResources+args.id+"/saved_model.pt")
    for w in weights:
        s = torch.seed()
        weights[w] = torch.rand(weights[w].size())
    # save R values
    torch.save(weights, args.pathToResources+args.id+"/"+args.id+"_random.pt")
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