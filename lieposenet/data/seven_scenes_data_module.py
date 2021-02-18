import pytorch_lightning as pl
import torch.utils.data
import torchvision.transforms as transforms

from .seven_scenes import SevenScenes


class SevenScenesDataModule(pl.LightningDataModule):
    def __init__(self, scene, data_path, batch_size=128, num_workers=4, split=(0.9, 0.1), seed=0, use_test=False,
                 image_size=256):
        super().__init__()
        torch.random.manual_seed(seed)
        image_transform = transforms.Compose([
            transforms.Resize(image_size),
            # transforms.CenterCrop(128),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        test_transform = transforms.Compose([
            transforms.Resize(image_size),
            # transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self._train_dataset = SevenScenes(scene, data_path, True, image_transform, mode=0, seed=seed)
        self._test_dataset = SevenScenes(scene, data_path, False, test_transform, mode=0, seed=seed)
        self._batch_size = batch_size
        self._num_workers = num_workers
        train_length = len(self._train_dataset)
        if use_test:
            self._train_subset, self._validation_subset = self._train_dataset, self._test_dataset
        else:
            lengths = int(train_length * split[0]), train_length - int(train_length * split[0])
            self._train_subset, self._validation_subset = torch.utils.data.random_split(self._train_dataset, lengths)
        print(f"[ToyDataModule] - train subset size {len(self._train_subset)}")
        print(f"[ToyDataModule] - validation dataset size {len(self._validation_subset)}")

    def train_dataloader(self, *args, **kwargs):
        return torch.utils.data.DataLoader(self._train_subset, self._batch_size, True, pin_memory=True,
                                           num_workers=self._num_workers)

    def val_dataloader(self, *args, **kwargs):
        return torch.utils.data.DataLoader(self._validation_subset, self._batch_size, False, pin_memory=True,
                                           num_workers=self._num_workers)

    def test_dataloader(self, *args, **kwargs):
        return torch.utils.data.DataLoader(self._test_dataset, self._batch_size, False, pin_memory=True,
                                           num_workers=self._num_workers)
