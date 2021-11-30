import argparse
import os
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

async def main():
    args = define_and_get_arguments()
    InterRes = torch.load(pathToResources+"/"+id+"/"+id+"_"+id+".pt")
    filelist = []
    with os.scandir(pathToResources+id) as dirs:
        for entry in dirs:
            if entry.name != (id+"_"+id+".npy") and re.search(".*\_"+id+".pt", entry.name):
                filelist.append(entry.name)
    for R in filelist:
        InterRes = InterRes + torch.load(pathToResources+"/"+id+"/"+R)
    torch.save(pathToResources+"/"+id+"/interRes.pt", InterRes)

if __name__ == "__main__":
    main()