from read_data import ImportData
from generate_partitions import PartitionGenerator
import argparse
import json
import os
import traceback

parser = argparse.ArgumentParser(description="Starting data partition...")
parser.add_argument("--read_from_json", help="first check json for parameters", action="store", default=True)
parser.add_argument("--datapath", help="pass path to data", action="store", default="../data")
parser.add_argument("--data_root", help="name of to data file", action="store", default=None)
parser.add_argument("--data_file_name", help="name of to data file", action="store", default="data.npy")
parser.add_argument("--target_file_name", help="name of targets file", action="store", default="target.npy")
parser.add_argument("--delim", help="specify in case csv format is used", action="store", default=",")
parser.add_argument("--target_name", help="name of the target column in whole dataset", action="store", default="target")
parser.add_argument("--data_name", help="list of nested features if applicable", action="store", default=None)
parser.add_argument("--partitions_num", help="number of partitions to generate", action="store", default=1)
parser.add_argument("--partitions_size", help="number of entities inside partition", action="store", default=1)
parser.add_argument("--save_to_directory", help="directory to save partition files", action="store", default='../learning/data/')
parser.add_argument("--data_prefix", help="prefix to use for data files", action="store", default='data')
parser.add_argument("--target_prefix", help="prefix to use for target files", action="store", default='target')
parser.add_argument("--shuffle", help="shuffle data before partitioning", action="store", default=True)
parser.add_argument("--partition_mode", help="random/equal/custom", action="store", default='random')
parser.add_argument("--data_asign", help="dictionary, storing a percent of each class entries for each client", action="store", default='random')

if __name__ == "__main__":
    args = parser.parse_args()
    if (args.read_from_json):
        with open('partition_config.json', 'rt') as f:
            t_args = argparse.Namespace()
            t_args.__dict__.update(json.load(f))
            args = parser.parse_args(namespace=t_args)
    
    print(args)
    data_file_name = str(args.datapath + '/' + args.data_file_name)
    if (not args.target_file_name == None):
        target_file_name = str(args.datapath + '/' + args.target_file_name)
    else:
        target_file_name = ""
    try:
        dataset = ImportData(data_file_name, target_file_name, target_name= args.target_name, delim = args.delim, 
                            data_name= args.data_name, data_root=args.data_root)
        partition_generator = PartitionGenerator(dataset)
        partition_generator.partition(args.partitions_num, args.partitions_size, 
        args.save_to_directory, args.data_prefix, args.target_prefix, args.shuffle, args.partition_mode, args.data_asign)
    except:
        traceback.print_exc()
        parser.print_help()
        input("Press Enter to continue...")