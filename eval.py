from argparse import ArgumentParser

import pytorch_lightning as pl
import torch

from lieposenet import ModelFactory
from lieposenet.data import SevenScenesDataModule
from lieposenet.utils import TensorBoardLogger, load_hparams_from_yaml
from lieposenet.utils.pose_net_result_evaluator import calculate_metrics
import json

parser = ArgumentParser(description="Evaluate pose net model")
parser.add_argument("--config", type=str, default="./configs/model.yaml")
parser.add_argument("--dataset_folder", type=str, default="./data/7scenes")
parser.add_argument("--dataset_name", type=str, default="chess")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--model", type=str, default="model.pth")
parser.add_argument("--result", type=str, default="metrics/metrics.json")
parser.add_argument("--data_saver_path", type=str, default="metrics/trajectories.npy")

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
trainer = pl.Trainer.from_argparse_args(arguments, logger=logger, deterministic=deterministic)

# Load parameters
params = load_hparams_from_yaml(arguments.config)
print("Load model from params \n" + str(params))

# Make data module
data_module_params = params.data_module
data_module = SevenScenesDataModule(arguments.dataset_name, arguments.dataset_folder,
                                    **data_module_params)

# Make model
model = ModelFactory().make_model(params.model, data_saver_path=arguments.data_saver_path)

# Load model
model.load_state_dict(torch.load(arguments.model)['state_dict'])

print("Start testing")
results = trainer.test(model, data_module.test_dataloader())
results[0].update(calculate_metrics(model._data_saver))

print("Final result:")
for key, value in results[0].items():
    print("{}: {}".format(key, value))

with open(arguments.result, "w") as fd:
    json.dump(results[0], fd)
