import argparse
import os
import numpy as np

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
    InterRes = np.load(pathToResources+id+"/"+id+"_"+id+".npy")
    filelist = []
    with os.scandir(pathToResources+id) as dirs:
        for entry in dirs:
            if entry.name != (id+"_"+id+".npy") and re.search(".*\_"+id+".npy", entry.name):
                filelist.append(entry.name)
    for R in filelist:
        InterRes = InterRes + np.load(pathToResources+id+"/"+R)
    np.save(pathToResources+id+"/"+"interRes.npy", InterRes)

if __name__ == "__main__":
    main()