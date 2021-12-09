import argparse
import numpy as np
import os
import sys
import asyncio
import torch
import json
from copy import deepcopy
from torchvision.models import vgg11
from model_configurations.simple_cnn import CNN
from model_configurations.mnist_model import MNIST
from pathlib import Path
import syft as sy
from torchvision import datasets, transforms
import warnings
warnings.simplefilter("ignore", np.RankWarning)

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
    parser.add_argument("--modelpath", default = 'saved_model_2')

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

async def main():
    #set up environment
    args = define_and_get_arguments()
    #os.chdir('./akka-server/Server/')
    torch.manual_seed(args.seed)
    print(args)
    hook = sy.TorchHook(torch)
    #define participants
    print(args.participantsjsonlist)

    #define model
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model, test_tensor = define_model(args.model_config, device, args.modelpath, int(args.model_output))
    #for p in model.parameters():
    #    p.register_hook(lambda grad: torch.clamp(grad, -6, 6))
    traced_model = torch.jit.trace(model,  test_tensor.to(device))
    traced_model.train()

    # extract interRes values
    publicKeys = json.loads(args.publicKeys.replace("=", ":"))
    interResList = {}
    for participant in publicKeys:
        interResList[participant] = {'publicKey': publicKeys[participant], 'weights': torch.load(args.pathToResources+"/interRes/"+participant+".pt")}

    # calculate aggregated weights
    aggregatedWeights = deepcopy(list(interResList.items())[0][1]['weights'])

    def setWeights(list0, lists, keys):  # recurrent calculations
        for i, x in enumerate(list0):
            if np.isscalar(x):
                list0[i] = np.polyfit(keys, np.array([list1[i] for list1 in lists]), int(args.degree))[-1]/len(interResList)
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

    """
    learning_rate = args.lr
    for curr_round in range(1, args.epochs + 1):
        learning_rate = max(0.98 * learning_rate, args.lr * 0.01)

        correct_predictions = 0
        all_predictions = 0
        traced_model.eval()
        results = test(worker_test, traced_model, args.batch_size, args.federate_after_n_batches, learning_rate, int(args.model_output))
        test_loss = []
        for curr_correct, total_predictions, loss, target_hist, predictions_hist in results:
            correct_predictions += curr_correct
            all_predictions += total_predictions
            test_loss.append(loss)
            print('Got predictions: \n')
            print(predictions_hist)
            print('Expected: \n')
            print(target_hist)

        print("Currrent accuracy: " + str(correct_predictions/all_predictions))
        print(test_loss)
    traced_model.train()
    """

    if args.modelpath:
        torch.save(model.state_dict(), args.modelpath)

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())