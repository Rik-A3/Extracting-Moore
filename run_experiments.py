import os
from experiments.train_transformers import run as train_run
from experiments.extract_automata import run as test_and_extract_run
import argparse

parser = argparse.ArgumentParser(prog='run_experiments')

parser.add_argument("--cuda_device", type=int, default=0, help="cuda device to use")
parser.add_argument("--project", default="extracting_fsms_test", help="save directories and logs to this wandb project")
parser.add_argument("--nb_workers", type=int, default=0, help="number of workers for dataloader")
parser.add_argument("--reps", type=int, default=1, help="number of repetitions")
parser.add_argument("--pos_embeddings", default="rotary", help="type of positional embeddings")
parser.add_argument("--l", type=int, default=32, help="length of sequences")
parser.add_argument("--N", type=int, default=10000, help="number of sequences")
parser.add_argument("--method", default="balanced", help="method of sampling")
parser.add_argument("--no_dups", action="store_true", help="no duplicates in dataset")
parser.add_argument("--dev", "-d", action="store_true", help="run in dev mode")
parser.add_argument("--run_id", type=str, default=None, help="run id to extract automata")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.cuda_device}"
os.environ["CUDA_USE_CUDA_DSA"] = "1"
if args.run_id is None:
    run_id = train_run(project=args.project, nb_workers=args.nb_workers, reps=args.reps, pos_embeddings=args.pos_embeddings,
            l=args.l, n=args.N, method=args.method, no_dups=args.no_dups, dev=args.dev)

else:
    run_id = args.run_id

test_and_extract_run(project=args.project, run_id=run_id, dev=args.dev, method=args.method, no_duplicates=args.no_dups)

