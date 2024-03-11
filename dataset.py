import lightning as L
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import CIFAR100
from torchvision import transforms

class CIFAR100(L.LightningDataModule):
    def __init__(self, data_dir: str = "./"):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                        std=(0.2023, 0.1994, 0.2010))
        ])

    def prepare_data(self):
        CIFAR100(root=self.data_dir, train=True, download=True)
        CIFAR100(root=self.data_dir, train=True, download=True)

    def setup(self, stage: str):
        if stage == "fit":
            cifar100_full = CIFAR100(self.data_dir, train=True, transform=self.transform)
            self.cifar100_train, self.cifar100_val = random_split(
                cifar100_full, [55000, 5000]
            )

            # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.cifar100_test = CIFAR100(self.data_dir, train=False, transform=self.transform)

        if stage == "predict":
            self.cifar100_predict = CIFAR100(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.cifar100_train, batch_size=32, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.cifar100_val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.cifar100_test, batch_size=32)

    def predict_dataloader(self):
        return DataLoader(self.cifar100_predict, batch_size=32)