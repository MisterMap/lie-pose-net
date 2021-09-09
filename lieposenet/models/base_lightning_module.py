import pytorch_lightning as pl
import torch


class BaseLightningModule(pl.LightningModule):
    def __init__(self, parameters, criterion):
        super().__init__()
        self.criterion = criterion
        self.save_hyperparameters(parameters)
        self._data_saver = None

    def loss(self, batch):
        raise NotImplementedError()

    def save_test_data(self, batch, output, losses):
        raise NotImplementedError()

    def metrics(self, batch, output):
        raise NotImplementedError()

    def additional_metrics(self):
        raise NotImplementedError()

    def training_step(self, batch, batch_index):
        output, losses = self.loss(batch)
        train_losses = {}
        for key, value in losses.items():
            train_losses[f"train_{key}"] = value
        self.log_dict(train_losses)
        return losses["loss"]

    def on_validation_epoch_start(self) -> None:
        self._data_saver.clear()

    def on_validation_epoch_end(self) -> None:
        metrics = self.additional_metrics()
        self.log_dict(metrics)

    def validation_step(self, batch, batch_index):
        output, losses = self.loss(batch)
        self.save_test_data(batch, output, losses)
        metrics = self.metrics(batch, output)
        val_losses = {}
        for key, value in losses.items():
            val_losses[f"val_{key}"] = value
        for key, value in metrics.items():
            val_losses[key] = value
        self.log_dict(val_losses)
        return losses["loss"]

    def test_step(self, batch, batch_index):
        output, losses = self.loss(batch)
        metrics = self.metrics(batch, output)
        self.save_test_data(batch, output, losses)
        val_losses = {}
        for key, value in losses.items():
            val_losses[f"test_{key}"] = value
        for key, value in metrics.items():
            val_losses[key] = value
        self.log_dict(val_losses)
        return losses["loss"]

    def configure_optimizers(self):
        if "betas" in self.hparams.optimizer.keys():
            beta1 = float(self.hparams.optimizer.betas.split(" ")[0])
            beta2 = float(self.hparams.optimizer.betas.split(" ")[1])
            self.hparams.optimizer.betas = (beta1, beta2)
        if self.criterion.lr is not None:
            criterion_parameters = self.criterion.parameters()
            other_parameters = set(self.parameters()) - set(criterion_parameters)
            learning_parameters = [
                {"params": list(other_parameters)},
                {"params": list(criterion_parameters), "lr": self.criterion.lr}
            ]
            print(learning_parameters)
        else:
            learning_parameters = self.parameters()
        print(learning_parameters)
        optimizer = torch.optim.Adam(learning_parameters, **self.hparams.optimizer)
        if "scheduler" in self.hparams.keys():
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **self.hparams.scheduler)
            return [optimizer], [scheduler]
        return optimizer

    @property
    def data_saver(self):
        return self._data_saver
