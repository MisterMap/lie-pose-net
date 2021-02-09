from argparse import ArgumentParser

import pytorch_lightning as pl

from lieposenet import ModelFactory
from lieposenet.data import SevenScenesDataModule
from lieposenet.utils import TensorBoardLogger, load_hparams_from_yaml

parser = ArgumentParser(description="Run Lie pose net")
parser.add_argument("--config", type=str, default="./configs/model.yaml")
parser.add_argument("--dataset_name", type=str, default="seven_scenes")
parser.add_argument("--dataset_folder", type=str, default="./data/7scenes")
parser.add_argument("--dataset_scene", type=str, default="chess")
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
params = load_hparams_from_yaml(arguments.config)
checkpoint_callback = pl.callbacks.ModelCheckpoint(filepath=arguments.out)
trainer = pl.Trainer.from_argparse_args(arguments, logger=logger, callbacks=[checkpoint_callback],
                                        deterministic=deterministic, max_epochs=params.max_epochs,
                                        profiler="simple")

# Make data module
data_module_params = params.data_module
data_module_params.scene = arguments.dataset_scene
data_module_params.data_path = arguments.dataset_folder
data_module_params.name = arguments.dataset_name
data_module_params.seed = arguments.seed
data_module = SevenScenesDataModule(data_module_params)

# Load parameters
model_params = params.model
print("Load model from params \n" + str(model_params))

# Make model
model = ModelFactory().make_model(model_params, train_dataset=data_module.get_train_dataset())

print("Start training")
trainer.fit(model, data_module)
