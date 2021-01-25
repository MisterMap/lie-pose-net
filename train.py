from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl

from lieposenet import ModelFactory
from lieposenet.utils import TensorBoardLogger, load_hparams_from_yaml
from lieposenet.data import SevenScenesDataModule

parser = ArgumentParser(description="Run Pose MVAE model")
parser.add_argument("--config", type=str, default="./configs/model.yaml")
parser.add_argument("--dataset_folder", type=str, default="./data/7scenes")
parser.add_argument("--dataset_name", type=str, default="chess")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--out", type=str, default="model.pth")

parser = pl.Trainer.add_argparse_args(parser)
arguments = parser.parse_args()

logger = TensorBoardLogger("lightning_logs")

# Seed
deterministic = False
seed = 0
if arguments.seed is not None:
    pl.seed_everything(arguments.seed)
    deterministic = True
    seed = arguments.seed

# Make trainer
checkpoint_callback = pl.callbacks.ModelCheckpoint(filepath=arguments.out)
trainer = pl.Trainer.from_argparse_args(arguments, logger=logger, callbacks=[checkpoint_callback],
                                        deterministic=deterministic)

# Make data module
data_module = SevenScenesDataModule(arguments.dataset_name, arguments.dataset_folder,
                                    arguments.batch_size, arguments.num_workers)

# Load parameters
params = load_hparams_from_yaml(arguments.config).model
print("Load model from params \n" + str(params))

# Make model
model = ModelFactory().make_model(params)

print("Start training")
trainer.fit(model, data_module)
