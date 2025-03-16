import os
import sys

import yaml

from experiments.train_transformers import run as train_run
from experiments.extract_automata import run as test_and_extract_run
from src.utils import set_seed

def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    os.environ["CUDA_VISIBLE_DEVICES"] = f"{config["cuda_device"]}"
    os.environ["CUDA_USE_CUDA_DSA"] = "1"
    set_seed(config["seed"])

    # train a model
    if config["run_id"] is None:
        run_id = train_run(project=config["project"], nb_workers=config["nb_workers"], reps=config["reps"],
                           pos_embeddings=config["pos_embeddings"],
                           l=config["l"], n=config["N"], method=config["method"], no_dups=config["no_dups"], dev=config["dev"])
    # load a model
    else:
        run_id = config["run_id"]

    test_and_extract_run(project=config["project"], run_id=run_id, dev=config["dev"], method=config["method"], no_duplicates=config["no_dups"])

if __name__ == "__main__":
    main(*sys.argv[1:])
    #main("configs/positive_only.yaml")