import numpy as np
from scipy import io as sio
import torch
import os
import imageio
import pandas as pd

class ImportData():

    def read_csv(self, data_path, delim, target_path, target_name):
        loaded_data = pd.read_csv(data_path, sep = delim)
        loaded_data = pd.DataFrame(loaded_data)
        if (os.path.isfile(target_path)):
                data = loaded_data
                targets = pd.read_csv(target_path,  sep = delim)
        else:
            try:
                targets = loaded_data[target_name].to_numpy()
            except:
                targets = np.empty(0)
            data = loaded_data.drop(target_name).to_numpy()
        return data, targets

    def read_images_from_folder(self, data_path):
        labels = os.listdir(data_path)
        targets = np.empty(0, np.str)
        data = []
        for label in labels:
            images = os.listdir(str(data_path + "/"+ label))
            for image in images:
                targets = np.append(targets, label)
                curr_image = imageio.imread(str(data_path + "/"+ label + "/" + image))
                if np.shape(curr_image)[2] == 4:
                    curr_image = self.convert_image(curr_image)
                data.append(curr_image)
        data = np.asarray(data) 
        return data, targets

    def read_matlab_file(self, data_path, data_root, data_name, target_path, target_name):
        loaded_data = sio.loadmat(data_path)
        if (not data_root == None):
            data_root = data_root.split(',')
            for name in data_root:
                name = name.strip()
                loaded_data = loaded_data[name]
                if np.shape(loaded_data) in [(1, 0), (1, 1), (1, ) ]:
                    loaded_data = loaded_data[0, 0]
        if (os.path.isfile(target_path)):
                targets = sio.loadmat(target_path)
        else:
            try:
                targets = loaded_data[target_name]
                if len(targets.shape) > 1:
                    targets = targets.flatten()  
            except:
                targets = np.empty(0)
        data = loaded_data[data_name]
        return data, targets

    def convert_to_tensor(self, data):
        data = data.permute(2, 1, 0)
        return data

    def convert_image(self, np_image):
        new_image = np.zeros((np_image.shape[0], np_image.shape[1], 3), dtype=np.uint8) 
        for each_channel in range(3):
            new_image[:,:,each_channel] = np_image[:,:,each_channel]  
        return new_image

    def __init__(self, data_path = '', target_path = '', target_name = 'target', delim = ',', 
         data_name = [], data_root = None):
        if (os.path.isdir(data_path)):
            self.data, self.targets = self.read_images_from_folder(data_path)
        else:
            filename_split_extension = os.path.splitext(data_path)
            datatype = filename_split_extension[1]
            if datatype == '.npy':
                self.data = np.load(data_path, allow_pickle=True)
                if (os.path.isfile(target_path)):
                    self.targets = np.load(target_path, allow_pickle=True)
                else:
                    raise Exception("File with targets is not found")
            if datatype == '.mat':
                self.data, self.targets = self.read_matlab_file(data_path, data_root, data_name, target_path, target_name)
            if datatype == '.csv':
                self.data, self.targets = self.read_csv(data_path, delim, target_path, target_name)
            if datatype == '.pt':
                self.data, self.targets = torch.load(data_path)
                self.data = [np.asarray(el) for el in self.data]
                self.data = np.asarray(self.data)
                self.targets = np.array(self.targets)
        
        unique, counts = np.unique(self.targets, return_counts=True)
        print(dict(zip(unique, counts)))
        self.data, self.targets


    def __len__(self):
        return len(self.targets)

        