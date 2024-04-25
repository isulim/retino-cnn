import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics as tm
from typing import Type

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from pathlib import Path


class LightningCNN(pl.LightningModule):
    """
    A Convolutional Neural Network (CNN) implemented using PyTorch Lightning.
    Loss function is BCELoss, optimizer is Adam.

    Parameters
    ----------
    conv_layers : int
        The number of convolutional layers.
    fc_layer_sizes : tuple of int
        The sizes of the fully connected layers.
    input_size : torch.Size
        The size of the input tensor.
    out_classes : int, optional
        The number of output classes, default is 1 (for binary classification).
    initial_filters : int, optional
        The number of filters in the first convolutional layer, default is 32.
    hl_kernel_size : int, optional
        The kernel size for the hidden layers, default is 5.
    activation_func : nn.Module, optional
        The activation function to use, default is nn.ReLU.
    max_pool_kernel : int, optional
        The kernel size for max pooling, default is 2.
    dropout_conv : bool, optional
        Whether to apply dropout to the convolutional layers, default is False.
    dropout_fc : bool, optional
        Whether to apply dropout to the fully connected layers, default is False.
    dropout_rate : float, optional
        The dropout rate, default is 0.5.
    initial_learning_rate : float, optional
        The initial learning rate, default is 0.01.
    """

    def __init__(
            self,
            *,
            conv_layers: int,
            fc_layer_sizes: tuple[int, ...],
            input_size: torch.Size,
            out_classes: int = 1,
            initial_filters: int = 32,
            hl_kernel_size: int = 5,
            hl_padding: int = 1,
            activation_func: Type[nn.Module] = nn.ReLU,
            max_pool_kernel: int = 2,
            dropout_conv: bool = False,
            dropout_fc: bool = False,
            dropout_rate: float = 0.5,
            initial_learning_rate: float = 0.01,
            loss_func: nn.Module = nn.BCEWithLogitsLoss(),
    ) -> None:

        # Validate inputs before calling super().__init__()
        self._validate_required_inputs(conv_layers, fc_layer_sizes, input_size)
        self._validate_default_inputs(
            out_classes,
            initial_filters,
            hl_kernel_size,
            max_pool_kernel,
            dropout_conv,
            dropout_fc,
            dropout_rate,
            initial_learning_rate,
        )
        super().__init__()

        device = "mps"

        # Initialize hyperparameters
        self._initial_learning_rate = initial_learning_rate

        self.loss_func = loss_func.to(device)

        # Initialize metrics
        self.accuracy = tm.Accuracy(task="binary").to(device)
        self.precision = tm.Precision(task="binary").to(device)
        self.recall = tm.Recall(task="binary").to(device)
        self.f1 = tm.F1Score(task="binary").to(device)
        self.auc = tm.AUROC(task="binary").to(device)
        self.confmat = tm.ConfusionMatrix(task="binary", num_classes=2).to(device)

        # Initialize convolutional layers}
        hidden_layers = []
        in_channels = input_size[0]

        for i in range(conv_layers):
            out_channels = initial_filters * 2 ** i
            hidden_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=hl_kernel_size, padding=hl_padding, device=device))
            hidden_layers.append(activation_func())
            hidden_layers.append(nn.MaxPool2d(max_pool_kernel))
            in_channels = out_channels
            if dropout_conv:
                hidden_layers.append(nn.Dropout(dropout_rate))

        self.hidden_layers = nn.Sequential(*hidden_layers)

        # Initialize fully connected layers
        in_features = self._get_conv_out_shape(input_size)
        fc_layers = [nn.Flatten()]
        for out_features in fc_layer_sizes:
            fc_layers.append(nn.Linear(in_features, out_features, device=device))
            fc_layers.append(activation_func())
            if dropout_fc:
                fc_layers.append(nn.Dropout(dropout_rate))
            in_features = out_features

        fc_layers.extend([
            nn.Linear(in_features, out_classes, device=device),
        ])

        self.fc_layers = nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor

        Returns
        -------
        torch.Tensor
            The output tensor
        """
        x = self.hidden_layers(x)
        x = self.fc_layers(x)
        return x

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Training step of the model.

        Parameters
        ----------
        batch : torch.Tensor
            The input batch
        batch_idx : int
            The index of the batch

        Returns
        -------
        torch.Tensor
            The loss
        """

        x, y = batch
        x = x.to("mps")
        y = y.to("mps")
        y = torch.unsqueeze(y, 1).float()
        y_pred = self(x)

        loss = self.loss_func(y_pred, y)

        self.accuracy(y_pred, y)
        self.log("train_accuracy", self.accuracy, on_epoch=True, on_step=False, prog_bar=True)

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Validation step of the model.

        Parameters
        ----------
        batch : torch.Tensor
            The input batch
        batch_idx : int
            The index of the batch

        Returns
        -------
        torch.Tensor
            The loss
        """
        x, y = batch
        x = x.to("mps")
        y = y.to("mps")
        y = torch.unsqueeze(y, 1).float()
        y_pred = self(x)

        loss = self.loss_func(y_pred, y)
        self.log("valid_loss", loss, on_epoch=True, on_step=False, prog_bar=True)

        self.accuracy(y_pred, y)
        self.precision(y_pred, y)
        self.recall(y_pred, y)
        self.f1(y_pred, y)
        self.auc(y_pred, y)
        self.confmat(y_pred, y)

        self.log("valid_accuracy", self.accuracy, on_epoch=True, on_step=False, prog_bar=True)
        self.log("valid_precision", self.precision, on_epoch=True, on_step=False, prog_bar=True)
        self.log("valid_recall", self.recall, on_epoch=True, on_step=False, prog_bar=True)
        self.log("valid_f1", self.f1, on_epoch=True, on_step=False, prog_bar=True)
        self.log("valid_auc", self.auc, on_epoch=True, on_step=False, prog_bar=True)

        return loss

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Test step of the model.

        Parameters
        ----------
        batch : torch.Tensor
            The input batch
        batch_idx : int
            The index of the batch

        Returns
        -------
        torch.Tensor
            The loss
        """
        x, y = batch
        x = x.to("mps")
        y = y.to("mps")
        y = torch.unsqueeze(y, 1).float()
        y_pred = self(x)
        loss = self.loss_func(y_pred, y)
        self.log("test_loss", loss, on_epoch=True, on_step=False, prog_bar=True)

        self.accuracy(y_pred, y)
        self.precision(y_pred, y)
        self.recall(y_pred, y)
        self.f1(y_pred, y)
        self.auc(y_pred, y)
        self.confmat(y_pred, y)

        self.log("test_accuracy", self.accuracy, on_epoch=True, on_step=False, prog_bar=True)
        self.log("test_precision", self.precision, on_epoch=True, on_step=False, prog_bar=True)
        self.log("test_recall", self.recall, on_epoch=True, on_step=False, prog_bar=True)
        self.log("test_f1", self.f1, on_epoch=True, on_step=False, prog_bar=True)
        self.log("test_auc", self.auc, on_epoch=True, on_step=False, prog_bar=True)

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure the optimizer for the model.

        Returns
        -------
        torch.optim.Optimizer
            The optimizer
        """
        return torch.optim.Adam(self.parameters(), lr=self._initial_learning_rate)

    def _get_conv_out_shape(self, input_size: torch.Size) -> torch.Tensor:
        """
        Calculate shape of the output of the convolutional layers.

        Parameters
        ----------
        input_size : torch.Size
            The size of the input tensor

        Returns
        -------
        torch.Size
            The size of the output tensor
        """
        with torch.no_grad():
            zeros = torch.zeros(*input_size, device="mps")
            z = self.hidden_layers(zeros)
            z = torch.prod(torch.tensor(z.shape))
        return z

    def _validate_required_inputs(self, conv_layers, fc_layer_sizes, input_size) -> None:
        """Validate inputs with no default values."""

        if not isinstance(conv_layers, int) or conv_layers < 1:
            raise ValueError("conv_layers must be an integer greater than 0.")

        if not isinstance(fc_layer_sizes, tuple) or not all(isinstance(i, int) for i in fc_layer_sizes):
            raise ValueError("fc_layer_sizes must be a tuple of integers.")

        if not isinstance(input_size, torch.Size):
            raise ValueError("input_size must be a torch.Size object.")

    def _validate_default_inputs(self, out_classes, initial_filters, hl_kernel_size, max_pool_kernel, dropout_conv,
                                 dropout_fc, dropout_rate, initial_learning_rate) -> None:
        """Validate inputs with default values."""

        if not isinstance(out_classes, int) or out_classes < 1:
            raise ValueError("out_classes must be an integer greater than 0.")

        if not isinstance(initial_filters, int) or initial_filters < 1:
            raise ValueError("initial_filters must be an integer greater than 0.")

        if not isinstance(hl_kernel_size, int) or hl_kernel_size < 1:
            raise ValueError("hl_kernel_size must be an integer greater than 0.")

        if not isinstance(max_pool_kernel, int) or max_pool_kernel < 1:
            raise ValueError("max_pool_kernel must be an integer greater than 0.")

        if not isinstance(dropout_conv, bool):
            raise ValueError("dropout_conv must be a boolean.")

        if not isinstance(dropout_fc, bool):
            raise ValueError("dropout_fc must be a boolean.")

        if not isinstance(dropout_rate, float) or not 0 <= dropout_rate <= 1:
            raise ValueError("dropout_rate must be a float between 0 and 1.")

        if not isinstance(initial_learning_rate, float) or initial_learning_rate <= 0:
            raise ValueError("initial_learning_rate must be a float greater than 0.")


class ImageDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str | Path, batch_size: int = 32, transformations: transforms.Compose = None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transformations = transformations
        if not transformations:
            self.transformations = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ])

    def setup(self, stage=None):
        self.train_dataset = ImageFolder(root=str(Path(self.data_dir, 'train')), transform=self.transformations)
        self.val_dataset = ImageFolder(root=str(Path(self.data_dir, 'val')), transform=self.transformations)
        self.test_dataset = ImageFolder(root=str(Path(self.data_dir, 'test')), transform=self.transformations)

    def train_dataloader(self):
        # class_sample_count = np.array([len(np.where(self.train_dataset.targets == t)[0]) for t in np.unique(self.train_dataset.targets)])
        # weights = 1. / class_sample_count
        # samples_weight = np.array([weights[t] for t in self.train_dataset.targets])
        # sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=8, persistent_workers=True,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=8, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=8, persistent_workers=True)
