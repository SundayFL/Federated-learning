import os
import argparse
from random import shuffle
import albumentations as A
import pickle 

import torch as th
from syft.workers.websocket_server import WebsocketServerWorker
from read_data import ImportData
from custom_dataset import CustomDataset
from torch.utils.data import Dataset, DataLoader
import syft as sy
import numpy as np
from torchvision import transforms
from PIL import Image
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
parser.add_argument("--data_file_name", help="name of to data file", action="store", default="data.npy")
parser.add_argument("--target_file_name", help="name of targets file", action="store", default="target.npy")
parser.add_argument("--data_set_id", type=int, help="id of data set", action="store", default= 1)
parser.add_argument("--model_config", type=str, help="chosen nn configuration", action="store", default='vgg')
parser.add_argument("--participantsjsonlist", help="show program version", action="store", default="{}")

possible_augmentations = A.Compose(
                            [A.Blur(blur_limit=3), 
                            A.GridDistortion(), 
                            A.OpticalDistortion(),
                            A.RandomRotate90(), 
                            A.HorizontalFlip(0.2),
                            A.GaussNoise(),
                            A.Sharpen()])

def augument_classes(data, target, mean_entries_per_class):
    new_entries = []
    new_targets = []
    entries_to_generate = int(mean_entries_per_class - len(target))
    for i in range(entries_to_generate):
        ind = np.random.choice(list(range(0, len(target))))
        new_entry = Image.fromarray(np.uint8(possible_augmentations(image=np.array(data[ind]))['image']))
        new_entries.append(new_entry)
        new_targets.append(target[0])
    return new_entries, new_targets

def get_transformation_seq(model_config):
    #predefined values to match model expectations
    transform_seq = []
    if (model_config == 'mobilenetv2'):
        transform_seq = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            
    if (model_config == 'cnn'):
        transform_seq = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()])
    
    if (model_config == 'mnist'):
        transform_seq = transforms.Compose([
            transforms.ToTensor()])

    if (transform_seq == []):
        raise Exception('Model config not found')
    return transform_seq

def main(details_dict, **kwargs):  # pragma: no cover
    """Helper function for spinning up a websocket participant."""
    #os.chdir("./akka-server/Client/src/main/modules")
    # Create websocket worker
    worker = WebsocketServerWorker(**kwargs)
    print(details_dict)
    if (details_dict["data_set_id"] == -1):
        data_file_name = str(details_dict["datapath"] + '/' + details_dict["data_file_name"])
        target_file_name = str(details_dict["datapath"] + '/' + details_dict["target_file_name"])
    else:
        print('Looking for partitioned sets with prefix... \n')
        data_prefix = os.path.splitext(str(details_dict["datapath"] + '/' + details_dict["data_file_name"]))[0]
        target_prefix = os.path.splitext(str(details_dict["datapath"] + '/' + details_dict["target_file_name"]))[0]
        data_file_name = str(data_prefix + '_' + str(details_dict["data_set_id"]) + '.npy')
        target_file_name= str(target_prefix + '_' + str(details_dict["data_set_id"]) + '.npy')

    dataset = ImportData(data_path = data_file_name, target_path = target_file_name)
    dataset.data = [Image.fromarray(np.uint8(im)) for im in dataset.data]
    unique_classes, counts = np.unique(dataset.targets, return_counts=True)
    dict_entries = dict(zip(unique_classes, counts))
    mean_entries_per_class = len(dataset.targets)/len(unique_classes)
    for image_class in unique_classes:
        index_target = np.squeeze(np.argwhere(dataset.targets == image_class))
        if dict_entries[image_class] < mean_entries_per_class and dict_entries[image_class] > 3:
            new_entries, new_targets = augument_classes([dataset.data[i] for i in index_target], [dataset.targets[i] for i in index_target], mean_entries_per_class)
            dataset.data = dataset.data + new_entries
            dataset.targets = np.concatenate((dataset.targets, new_targets))

    unique_classes, counts = np.unique(dataset.targets, return_counts=True)
    transformation_seq = get_transformation_seq(details_dict["model_config"])
    train_base = CustomDataset(imported_data=dataset, 
    transform = transformation_seq)
    train_base.targets = th.tensor(train_base.targets, dtype = th.int64)
    # check if tensors have correct dims
    sizes = list(map(np.shape, train_base.data))
    print(sizes[0])

    # Tell the worker about the dataset
    worker.add_dataset(train_base, key="mnist")
    worker.serializer
    # Start worker

    worker.start()

    return worker


if __name__ == "__main__":
    hook = sy.TorchHook(th)

    args = parser.parse_args()
    print(args)

    kwargs = {
        "id": args.id,
        "host": args.host,
        "port": args.port,
        "hook": hook,
        "verbose": args.verbose,
    }

    details_dict = {
        "data_set_id": args.data_set_id, 
        "data_file_name": args.data_file_name,
        "target_file_name": args.target_file_name,
        "datapath": args.datapath, 
        "model_config": args.model_config
    }

    main(details_dict, **kwargs)
