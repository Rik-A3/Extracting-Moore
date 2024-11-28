import os
import random
import torch
import wandb

from src.TargetMachines import get_target_moore
from src.generate_data import MooreDataset, get_dataset, get_dataset_name

from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from lightning.pytorch.loggers import WandbLogger
import torchmetrics

import pytorch_lightning as pl

from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from src.MooreOracle import MooreOracle, TransformerOracle

from transformer_lens import HookedTransformerConfig

from src.utils import AverageCorrectLength, Last10Accuracy, MinimalCorrectLength, SequenceAccuracy

class LightningModule(pl.LightningModule):
    def __init__(self, model : MooreOracle, learning_rate: float, nb_classes : int, seq_length=None, dev=False, wandb_test_group="test"):
        super().__init__()

        self.dev = dev

        self.seq_length=seq_length

        self.model = model
        self.loss = CrossEntropyLoss()
        self.learning_rate = learning_rate

        self.accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=nb_classes)
        self.test_accuracy = Last10Accuracy(nb_classes=nb_classes)
        self.sequence_accuracy = SequenceAccuracy(nb_classes=nb_classes)
        self.average_correct_length = AverageCorrectLength()
        self.minimal_correct_length = MinimalCorrectLength()

        self.wandb_test_group = wandb_test_group

    def setup(self, stage):
        self.accuracy.to(self.device)
        self.sequence_accuracy.to(self.device)

        return super().setup(stage)

    # For inference
    def forward(self, x) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch                # [batch_size * length]
        logits = self.forward(x)    # [batch_size * length * vocab_out]

        logits_flat = torch.flatten(logits, start_dim=0, end_dim=1)
        y_flat = torch.flatten(y, start_dim=0, end_dim=1)

        loss = self.loss(logits_flat, y_flat)

        y_pred = torch.argmax(torch.softmax(logits, dim=-1), axis=-1)
        y_pred_flat = torch.flatten(y_pred, start_dim=0, end_dim=1)

        self.log('loss', loss, prog_bar=True)
        self.log('train/acc', self.accuracy(y_pred_flat, y_flat), prog_bar=True)

        if not self.dev:
            wandb.log({'train/loss': loss,
                        'train/acc': self.accuracy(y_pred_flat, y_flat),
                        'train/seq_acc': self.sequence_accuracy(y_pred, y),
                        'train/avg_correct_length': self.average_correct_length(y_pred, y),
                        'train/min_correct_length': self.minimal_correct_length(y_pred, y),
                        })

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch                # [batch_size * length]
        logits = self.forward(x)    # [batch_size * length * vocab_out]

        logits_flat = torch.flatten(logits, start_dim=0, end_dim=1)
        y_flat = torch.flatten(y, start_dim=0, end_dim=1)

        loss = self.loss(logits_flat, y_flat)

        y_pred = torch.argmax(torch.softmax(logits, dim=-1), axis=-1)
        y_pred_flat = torch.flatten(y_pred, start_dim=0, end_dim=1)

        self.log('val/loss', loss, prog_bar=True)

        if not self.dev:
            wandb.log({'val/loss': loss,
                        'val/acc': self.accuracy(y_pred_flat, y_flat),
                        'val/seq_acc': self.sequence_accuracy(y_pred, y),
                        'val/avg_correct_length': self.average_correct_length(y_pred, y),
                        'val/min_correct_length': self.minimal_correct_length(y_pred, y),
                        })


    def test_step(self, batch, batch_idx):
        x, y = batch                # [batch_size * length]

        logits = self.forward(x)    # [batch_size * length * vocab_out]

        logits_flat = torch.flatten(logits, start_dim=0, end_dim=1)
        y_flat = torch.flatten(y, start_dim=0, end_dim=1)

        loss = self.loss(logits_flat, y_flat)

        y_pred = torch.argmax(torch.softmax(logits, dim=-1), axis=-1)

        self.log(f'{self.wandb_test_group}/loss', loss, prog_bar=True)

        if not self.dev:
            self.test_accuracy.to(self.device)
            wandb.log({f'{self.wandb_test_group}/loss-{self.seq_length}': loss,
                        f'{self.wandb_test_group}/acc-{self.seq_length}': self.test_accuracy(y_pred, y),
                        f'{self.wandb_test_group}/seq_acc-{self.seq_length}': self.sequence_accuracy(y_pred, y),
                        f'{self.wandb_test_group}/avg_correct_length-{self.seq_length}': self.average_correct_length(y_pred, y),
                        f'{self.wandb_test_group}/min_correct_length-{self.seq_length}': self.minimal_correct_length(y_pred, y),
                        })

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def save_model(self, name):
        torch.save(self.model, name+".pt")

def get_wandb_logger(cfg):
    print(cfg["name"])
    return WandbLogger(name=cfg["name"], project=cfg['project_name'], config=cfg, log_model=False)

def train_run(cfg, train_dataset: MooreDataset, val_dataset: MooreDataset, rep=0, dev=False, nb_workers=0):
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    model = TransformerOracle(cfg=cfg).to(device)
    pl_module = LightningModule(model, cfg["lr"], nb_classes=cfg["d_vocab_out"], dev=dev).to(device)

    early_stop_callback = EarlyStopping(monitor="val/loss", min_delta=0.00, patience=cfg["patience"])
    trainer = pl.Trainer(accelerator=device.type,
                         devices=1, 
                         max_epochs=cfg["epochs"], 
                         logger=(None if dev else get_wandb_logger(cfg)), 
                         detect_anomaly=dev,
                         fast_dev_run=dev, 
                         callbacks=[early_stop_callback])

    train_dataloader = DataLoader(train_dataset, batch_size=cfg["batch_size"], num_workers=nb_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg["batch_size"], num_workers=nb_workers)

    trainer.fit(pl_module, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    
    if not os.path.exists("models/"+cfg["project_name"]):
        os.makedirs("models/"+cfg["project_name"])
    pl_module.save_model("models/"+cfg["project_name"]+f"/{cfg["name"]}")

    wandb.finish()

def train_model(train_cfg, language_name, length, N, val_length, val_N, repetitions=5, method="random", val_method="random", no_duplicates=True, val_no_duplicates=True, dev = False, nb_workers=0):
    if not os.path.exists(train_cfg["project_name"]):
        os.mkdir(train_cfg["project_name"])
    
    # Load data
    train_dataset = get_dataset(language_name, length, N, train=True, method=method, no_duplicates=no_duplicates)
    print("Train shape:", train_dataset.data.shape)
    val_dataset = get_dataset(language_name, val_length, val_N, train=False, method=val_method, no_duplicates=val_no_duplicates)

    unformatted_name = train_cfg["name"]
    print(unformatted_name)
    for r in range(repetitions):
        train_cfg.update({"name": unformatted_name.format(r), "repetition": r})
        train_run(train_cfg, train_dataset, val_dataset, rep=r, dev = dev, nb_workers=nb_workers)

WANDB_PROJECT = "Extracting-test"
BATCH_SIZE = 32
NB_WORKERS = 0
REPS = 1

N_LAYERS = 1
D_MODEL = 16
D_MLP = 64
D_HEADS = 4
POS_EMBEDDINGS = "rotary"

L = 12
N = 10_000
METHOD = "random"
NO_DUPS = True

VAL_L = 100
VAL_N = 1_000
VAL_METHOD = "random"
VAL_NO_DUPS = True

def run(project=WANDB_PROJECT, batch_size=BATCH_SIZE, nb_workers=NB_WORKERS, reps=REPS, 
        n_layers=N_LAYERS, d_model= D_MODEL, d_mlp=D_MLP, d_heads=D_HEADS, pos_embeddings=POS_EMBEDDINGS,
        l=L, n=N, val_l=VAL_L, val_N=VAL_N, method=METHOD, val_method=VAL_METHOD, no_dups=NO_DUPS, val_no_dups=VAL_NO_DUPS, dev=False):

    languages = [
        "ones",
        "dyck_1",
        "dyck_2",
        "grid_1",
        "parity",
        "grid_2",
        "first",
    ]
    
    run_id = random.randint(0, 10_000_000)  # identifier for this run

    for language in languages:
        target_machine = get_target_moore(language, training_task=method)

        train_cfg = {
            "project_name": project, 
            "run_id": run_id,
            "name":f"{language}-{run_id}"+"-{}",
            "language_name": language,
            "dataset_name": get_dataset_name(language, l, n, method, no_dups, train=True, data=True),
            "criterion":"BCE", 
            "epochs":10_000, 
            "batch_size": batch_size,
            "lr":3e-4,
            "n_layers" : n_layers,
            "d_model": d_model,
            "d_mlp": d_mlp,
            "n_ctx" : 1024,
            "d_head" : d_heads,    
            "n_heads" : d_model // d_heads,
            "act_fn":'relu',
            "d_vocab": len(target_machine.alphabet)+1,                   # +1 for BOS token
            "d_vocab_out": max([target_machine.X[q] for q in target_machine.Q])+1,   # +1 bcs start at zero
            "weights": None,
            "patience": 5,
            "lr": 3e-4,
            "method": method,
            "normalization_type": "LNPre",
            "positional_embedding_type": pos_embeddings,
            "warmup_steps": 80,
        }

        if method == "character_prediction":
            train_cfg["d_vocab_out"] = len(target_machine.alphabet)**2

        train_model(train_cfg, language, l, n, val_l, val_N, repetitions=reps, method=method, val_method=val_method, dev=dev, nb_workers=nb_workers, no_duplicates=no_dups, val_no_duplicates=val_no_dups)

    return run_id