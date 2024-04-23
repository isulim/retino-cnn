import pytorch_lightning as pl
import torch.nn as nn
import torchmetrics as tm
import torchvision.models as models

from torch.optim import SGD, Adam, AdamW
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


class ResNetClassifier(pl.LightningModule):
    resnets = {
        18: models.resnet18,
        34: models.resnet34,
        50: models.resnet50,
        101: models.resnet101,
        152: models.resnet152,
    }

    optimizers = {
        "adam": Adam,
        "adamw": AdamW,
        "sgd": SGD,
    }

    def __init__(
            self,
            num_classes,
            resnet_version,
            train_path,
            val_path,
            test_path=None,
            optimizer="adam",
            lr=1e-3,
            batch_size=16,
            tune_fc_only=True,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.lr = lr
        self.batch_size = batch_size

        self.optimizer = self.optimizers[optimizer]
        self.loss_fn = (nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss())
        # Metrics
        self.accuracy = tm.Accuracy(task="binary").to(self.device)

        # Using a pretrained ResNet backbone
        self.resnet_model = self.resnets[resnet_version]()

        # Replace old FC layer with Identity to train own
        linear_size = list(self.resnet_model.children())[-1].in_features

        # replace final layer for fine-tuning
        self.resnet_model.fc = nn.Linear(linear_size, num_classes)

        if tune_fc_only:  # option to only tune the fully-connected layers
            for child in list(self.resnet_model.children())[:-1]:
                for param in child.parameters():
                    param.requires_grad = False

        self.save_hyperparameters()

    def forward(self, X):
        return self.resnet_model(X)

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr)

    def _step(self, batch):
        x, y = batch
        preds = self(x)

        if self.num_classes == 1:
            preds = preds.flatten()
            y = y.float()

        loss = self.loss_fn(preds, y)
        acc = self.accuracy(preds, y)
        return loss, acc

    def _dataloader(self, data_path, shuffle=False):
        transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )
        img_folder = ImageFolder(data_path, transform=transform)

        return DataLoader(img_folder, batch_size=self.batch_size, shuffle=shuffle, num_workers=8,
                          persistent_workers=True)

    def train_dataloader(self):
        return self._dataloader(self.train_path, shuffle=True)

    def training_step(self, batch, batch_idx):
        loss, acc = self._step(batch)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def val_dataloader(self):
        return self._dataloader(self.val_path)

    def validation_step(self, batch, batch_idx):
        loss, acc = self._step(batch)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True, logger=True)

    def test_dataloader(self):
        return self._dataloader(self.test_path)

    def test_step(self, batch, batch_idx):
        loss, acc = self._step(batch)

        self.log("test_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_acc", acc, on_epoch=True, prog_bar=True, logger=True)