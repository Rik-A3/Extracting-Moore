import os
import wandb
import torch
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformerConfig

import pytorch_lightning as pl

from src.MooreOracle import TransformerOracle
from src.generate_data import MooreDataset, label_data, get_dataset
from src.TargetMachines import accepting_states
from src.utils import get_model_paths, tensor_to_str_list

from extraction import extract
from experiments.train_transformers import LightningModule, get_wandb_logger



L_TEST = 32
N_TEST = 1_000
METHOD_TEST = "balanced"
NO_DUPS_TEST = False

TIME_LIMIT = 30
INIT_DEPTH = 10

def extract_run(model: TransformerOracle, starting_examples=[], time_limit=TIME_LIMIT, initial_depth=INIT_DEPTH):        
    cfg = model.full_cfg

    model.to("cpu")

    moore, time = extract(model, time_limit, initial_depth, starting_examples)

    if not os.path.exists("automata"):
        os.mkdir("automata")
    if not os.path.exists(f"automata/{cfg["project_name"]}"):
        os.mkdir(f"automata/{cfg["project_name"]}")
    moore.save(f"{cfg["project_name"]}/{cfg["run_id"]}")

    moore.draw_nicely(maximum=30, name=model.full_cfg["name"])


    return moore, time


def get_starting_examples(language):
    starting_examples_dict = {k : ["", "0", "1", "01", "10"] for k in accepting_states.keys()}
    starting_examples_dict["length_2"] = ["", "0", "00"]
    starting_examples_dict["length_3"] = ["", "0", "00"]
    starting_examples_dict["length_4"] = ["", "0", "00"]

    if language in starting_examples_dict:  
        return starting_examples_dict[language]

    return []

def test_run(model: TransformerOracle, test_dataloader, length=L_TEST, wandb_group="test", device="cuda"):
    cfg: HookedTransformerConfig = model.full_cfg
    logger = get_wandb_logger(cfg)

    pl_module = LightningModule(model, learning_rate=0, nb_classes=cfg["d_vocab_out"], seq_length=length, wandb_test_group=wandb_group).to(device)
    trainer = pl.Trainer(accelerator=device.type,
                        devices=1, 
                        max_epochs=1, 
                        logger=logger)
    
    trainer.test(pl_module, dataloaders=test_dataloader)

def run(project, run_id, dev=False, length=L_TEST, N=N_TEST, method=METHOD_TEST, no_duplicates=NO_DUPS_TEST):
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    for model_path in get_model_paths(project, run_id):
        model = torch.load(model_path)
        if not dev:
            cfg = model.full_cfg
            name = cfg["name"]
            wandb.init(project=project, name=name, config=cfg, resume="allow")

        test_dataset = get_dataset(cfg["language_name"], length, N, train=False, test=True, method=method, no_duplicates=no_duplicates, device="cpu")
        test_dataloader = DataLoader(test_dataset, batch_size=cfg["batch_size"], num_workers=2)

        test_run(model, test_dataloader, length=length, device=device)

        starting_examples = get_starting_examples(model.full_cfg["language_name"])
        
        moore, time = extract_run(model, starting_examples=starting_examples)

        wandb.log({"time": time})

        data = test_dataset.data
        data_lst = tensor_to_str_list(data)
        data_lst = [d[1:] for d in data_lst]    # remove bos token
        moore_labels = label_data(data_lst, moore)
        moore_labels = [[moore.X[moore.q0]] + d for d in moore_labels]    # add bos label
        moore_labels = torch.tensor(moore_labels, dtype=torch.long)   # [ batch_size * seq_len ]
        
        test_dataset = MooreDataset(data, moore_labels, device="cpu")
        test_dataloader = DataLoader(test_dataset, batch_size=cfg["batch_size"], num_workers=2)
        model.to(device)
        test_run(model, test_dataloader, length=length, wandb_group="extract", device=device)

        wandb.finish()








    
