import pytorch_lightning as pl
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np


class CelebADataDictWrapper(Dataset):
    def __init__(self, dset):
        super().__init__()
        self.dset = dset

    def __getitem__(self, i):
        x, y = self.dset[i]
        return {"jpg": x, "cls": y}

    def __len__(self):
        return len(self.dset)

class CelebALoader(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers=0, image_size=256, shuffle=True):
        super().__init__()
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2.0 - 1.0)
        ])
        self.batch_size = batch_size
        self.num_workers = 0#num_workers
        self.shuffle = shuffle
        self.image_size = image_size
        self.train_dataset = CelebADataDictWrapper(
            torchvision.datasets.CelebA(
                root="../data", split="train", target_type="attr", download=False, transform=transform
            )
        )
        self.val_dataset = CelebADataDictWrapper(
            torchvision.datasets.CelebA(
                root="../data", split="valid", target_type="attr", download=False, transform=transform
            )
        )
        self.test_dataset = CelebADataDictWrapper(
            torchvision.datasets.CelebA(
                root="../data", split="test", target_type="attr", download=False, transform=transform
            )
        )

    def prepare_data(self):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        ) 