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
from copy import deepcopy
import os
from torchvision.models import vgg11
from model_configurations.simple_cnn import CNN
from model_configurations.mnist_model import MNIST

import syft as sy
from syft.workers import websocket_client
from syft.frameworks.torch.fl import utils
import warnings
warnings.simplefilter("ignore", np.RankWarning)


websocket_client.TIMEOUT_INTERVAL = 60
LOG_INTERVAL = 25

logger = logging.getLogger("run_websocket_client")


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
    parser.add_argument("--pathToResources", help="where to store", action="store")
    parser.add_argument("--publicKeys", help="public keys", action="store")
    parser.add_argument("--degree", help="public keys", action="store")
    parser.add_argument("--participantsjsonlist", help="show program version", action="store", default="{}")
    parser.add_argument("--epochs", type=int, help="show program version", action="store", default=10)
    parser.add_argument("--model_config", default="vgg")
    parser.add_argument("--model_output", default=12)
    parser.add_argument("--modelpath", default = 'saved_model')

    args = parser.parse_args(args=args)
    return args

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

async def test(test_worker, traced_model, batch_size, federate_after_n_batches, learning_rate, model_output):

    model_config = sy.TrainConfig(
        model=traced_model,
        loss_fn=loss_fn,
        batch_size=batch_size,
        shuffle=True,
        max_nr_batches=federate_after_n_batches,
        epochs=1,
        optimizer="Adam",
        optimizer_args={"lr": learning_rate, "weight_decay": learning_rate*0.1},
    )
    with torch.no_grad():
        model_config.send(test_worker)
        worker_result = test_worker.evaluate(dataset_key="mnist", return_histograms = True, nr_bins = model_output)
    return worker_result['nr_correct_predictions'], worker_result['nr_predictions'], worker_result['loss'], worker_result['histogram_target'],  worker_result['histogram_predictions']

def define_participants_lists(participantsjsonlist, **kwargs_websocket):

    # choose at least 30% participants that will test the models

    participants = participantsjsonlist.replace("'","\"")
    participants = json.loads(participants)

    for_test = random.choices(participants, k=np.int(np.round(len(participants)*0.3)))
    print('Clients chosen for test: \n')
    print(for_test)
    worker_instances = []
    worker_instances_test = []
    for participant in participants:
        print("----------------------")
        print(participant['id'])
        print(participant['port'])
        if participant not in for_test:
            worker_instances.append(sy.workers.websocket_client.WebsocketClientWorker(id=participant['id'], port=participant['port'], **kwargs_websocket))
        else:
            worker_instances_test.append(sy.workers.websocket_client.WebsocketClientWorker(id=participant['id'], port=participant['port'], **kwargs_websocket))
    print("----------------------")

    for wcw in worker_instances:
        wcw.clear_objects_remote()

    for wcw in worker_instances_test:
        wcw.clear_objects_remote()

    return worker_instances, worker_instances_test

async def main():
    #set up environment
    args = define_and_get_arguments()
    #os.chdir('./akka-server/Server/')
    torch.manual_seed(args.seed)
    print(args)
    hook = sy.TorchHook(torch)
    kwargs_websocket = {"hook": hook, "verbose": args.verbose, "host": 'localhost'}
    #define participants
    worker_instances, worker_instances_test = define_participants_lists(args.participantsjsonlist, **kwargs_websocket)

    #define model
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model, test_tensor = define_model(args.model_config, device, args.modelpath, int(args.model_output))
    #for p in model.parameters():
    #    p.register_hook(lambda grad: torch.clamp(grad, -6, 6))

    # extract interRes values
    publicKeys = json.loads(args.publicKeys.replace("=", ":"))
    interResList = {}
    for participant in publicKeys:
        interResList[participant] = {'publicKey': publicKeys[participant], 'weights': torch.load(args.pathToResources+"/interRes/"+participant+".pt")}
        if os.path.exists(args.pathToResources+"/interRes/"+participant+".pt"):
            os.remove(args.pathToResources+"/interRes/"+participant+".pt")
    os.rmdir(args.pathToResources+"/interRes")

    # calculate aggregated weights
    aggregatedWeights = deepcopy(list(interResList.items())[0][1]['weights'])

    def setWeights(list0, lists, keys):  # recurrent calculations
        for i, x in enumerate(list0):
            if np.isscalar(x):
                list0[i] = np.polyfit(keys, np.array([list1[i] for list1 in lists]), int(args.degree))[-1]/len(interResList)
                # polynomial fit
            else:
                list0[i] = setWeights(list0[i], [list1[i] for list1 in lists], keys)
        return list0

    for weighttensor in aggregatedWeights:
        aggregatedWeights[weighttensor] = setWeights(
            np.array(aggregatedWeights[weighttensor]),
            np.array([np.array(interResList[interRes]['weights'][weighttensor]) for interRes in interResList]),
            np.array([interResList[interRes]['publicKey'] for interRes in interResList])
        )
        aggregatedWeights[weighttensor] = torch.tensor(aggregatedWeights[weighttensor])
    model.load_state_dict(aggregatedWeights)  # fit the model's structure

    model = torch.jit.trace(model, test_tensor.to(device))
    model.train()
    # model testing

    learning_rate = args.lr
    correct_predictions = 0
    all_predictions = 0
    model.eval()
    if len(worker_instances_test) > 0:
        results = await asyncio.gather(
            *[
                test(worker_test, model,
                     args.batch_size,
                     args.federate_after_n_batches, learning_rate, int(args.model_output))
                for worker_test in worker_instances_test
            ]
        )
        test_loss = []
        for curr_correct, total_predictions, loss , target_hist, predictions_hist in results:
            correct_predictions += curr_correct
            all_predictions += total_predictions
            test_loss.append(loss)
            print('Got predictions: \n')
            print(predictions_hist)
            print('Expected: \n')
            print(target_hist)

        print("Currrent accuracy: " + str(correct_predictions/all_predictions))
        print(test_loss)
    model.train()

    if args.modelpath:
        torch.save(model.state_dict(), args.modelpath)

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