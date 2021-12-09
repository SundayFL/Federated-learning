import argparse
import os
import sys
import re
import asyncio
import numpy as np
import torch

def define_and_get_arguments(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description="Calculate InterRes values."
    )
    parser.add_argument("--pathToResources", help="pass path to resources", action="store")
    parser.add_argument(
        "--id", type=str, action="store", help="name (id) of the client"
    )

    args = parser.parse_args(args=args)
    return args

def main():
    args = define_and_get_arguments()
    id = args.id
    pathToResources = args.pathToResources
    InterRes = torch.load(pathToResources+"/"+id+"/"+id+"_"+id+".pt")
    filelist = []
    with os.scandir(pathToResources+id) as dirs:
        for entry in dirs:
            if entry.name != (id+"_"+id+".npy") and re.search(".*\_"+id+".pt", entry.name):
                filelist.append(entry.name)
    for R in filelist:
        nextR = torch.load(pathToResources+"/"+id+"/"+R)
        for tensor in nextR:
            InterRes[tensor] = InterRes[tensor] + nextR[tensor]
    print(pathToResources+id+"/interRes.pt")
    torch.save(InterRes, pathToResources+id+"/interRes.pt")

if __name__ == "__main__":
    main()