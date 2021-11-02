import numpy as np
import pandas as pd
import random
import json

class PartitionGenerator:

    def __init__(self, dataset, partition_column = None):
        self.dataset = dataset
        self.partition_column = partition_column

        self

    def partition_equally(self, partition_min_size, partition_number):
        unique_classes = np.unique(self.dataset.targets)
        entities_per_partition = np.int(np.round(partition_min_size / len(unique_classes), decimals=0))
        data = [[] for y in range(partition_number)] 
        targets = [[] for y in range(partition_number)] 
        for category in unique_classes:
            selected_index = np.squeeze(np.argwhere(self.dataset.targets == category))
            for i in range(partition_number):
                if (i+1)*entities_per_partition-1 < len(selected_index):
                    current_index = selected_index[range(i*entities_per_partition,((i+1)*entities_per_partition))]
                else:
                    current_index = selected_index[i*entities_per_partition:]
                [data[i].append(arr) for arr in self.dataset.data[current_index]]
                targets[i] = np.concatenate((targets[i], self.dataset.targets[current_index]), axis=0)
        return data, targets

    def partition_custom(self, partition_number, partition_min_size, data_asign):
        unique_classes = np.unique(self.dataset.targets)
        data_asign = json.loads(data_asign.replace("'","\""))
        data = [[] for y in range(partition_number)] 
        targets = [[] for y in range(partition_number)] 
        indexes_available = []
        for category in range(len(unique_classes)):
            selected_index = np.squeeze(np.argwhere(self.dataset.targets == category))
            indexes_available.append(selected_index)
        for i in range(partition_number):
            entities_per_class = np.round(np.array(data_asign[0][str(i)], dtype=np.float)*partition_min_size, decimals=0)
            for category in range(len(unique_classes)):
                if (entities_per_class[category] > 0):
                    if entities_per_class[category] < len(indexes_available[category]):
                        current_index = indexes_available[category][range(0, np.int(entities_per_class[category]))]
                    else:
                        current_index = indexes_available[category]
                    [data[i].append(arr) for arr in self.dataset.data[current_index]]
                    targets[i] = np.concatenate((targets[i], self.dataset.targets[current_index]), axis=0)
                    indexes_available[category] = np.delete(indexes_available[category], range(0, np.int(entities_per_class[category])))
        return data, targets

    def partition(self, partition_number, partition_min_size, dir_save, data_prefix, target_prefix,
        shuffle, partition = 'random', data_asign = None):
        self.dataset.targets = np.array(pd.array(self.dataset.targets).astype('category').codes)
        
        if (shuffle):
            idx = np.arange(len(self.dataset.targets))
            random.shuffle(idx)
            self.dataset.data = self.dataset.data[np.argsort(idx)]
            self.dataset.targets = self.dataset.targets[np.argsort(idx)]

        if partition == 'random':
            data = np.split(self.dataset.data[range(0, partition_number*partition_min_size )], partition_number)
            targets = np.split(self.dataset.targets[range(0, partition_number*partition_min_size)], partition_number)
        
        if partition == 'equal':
            data, targets = self.partition_equally(partition_min_size, partition_number)

        if partition == 'custom':
            data, targets = self.partition_custom(partition_number, partition_min_size, data_asign)

        data = np.asarray(data)
        targets = np.asarray(targets)

        for partition_num in range(0, partition_number):
            np.save('./' + dir_save + '/' +data_prefix + '_' + str(partition_num) + '.npy', data[partition_num], allow_pickle= True)
            np.save('./' + dir_save + '/' +target_prefix + '_' + str(partition_num) + '.npy', targets[partition_num], allow_pickle= True)

