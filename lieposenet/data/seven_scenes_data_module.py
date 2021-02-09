import pytorch_lightning as pl
import torch.utils.data
import torchvision.transforms as transforms

from .dataset_factory import DatasetFactory


class SevenScenesDataModule(pl.LightningDataModule):
    def __init__(self, params, split=(0.9, 0.1)):
        super().__init__()
        torch.random.manual_seed(params.seed)
        image_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self._train_dataset = DatasetFactory().make_dataset(params, train=True, transform=image_transform)
        self._test_dataset = DatasetFactory().make_dataset(params, train=False, transform=image_transform)
        self._batch_size = params.batch_size
        self._num_workers = params.num_workers
        train_length = len(self._train_dataset)
        if params.use_test:
            self._train_subset, self._validation_subset = self._train_dataset, self._test_dataset
        else:
            lengths = int(train_length * split[0]), train_length - int(train_length * split[0])
            self._train_subset, self._validation_subset = torch.utils.data.random_split(self._train_dataset, lengths)
        print(f"[ToyDataModule] - train subset size {len(self._train_subset)}")
        print(f"[ToyDataModule] - validation dataset size {len(self._validation_subset)}")

    def get_train_dataset(self):
        return self._train_dataset

    def train_dataloader(self, *args, **kwargs):
        return torch.utils.data.DataLoader(self._train_subset, self._batch_size, True, pin_memory=True,
                                           num_workers=self._num_workers)

    def val_dataloader(self, *args, **kwargs):
        return torch.utils.data.DataLoader(self._validation_subset, self._batch_size, False, pin_memory=True,
                                           num_workers=self._num_workers)

    def test_dataloader(self, *args, **kwargs):
        return torch.utils.data.DataLoader(self._test_dataset, self._batch_size, False, pin_memory=True,
                                           num_workers=self._num_workers)
