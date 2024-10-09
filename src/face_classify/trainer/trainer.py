import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import lightning as L
from face_classify.models.build_model import Classifier

class Trainer(L.LightningModule):
    def __init__(
        self, num_classes=6, learning_rate=0.001
    ):
        super().__init__()
        self.model = Classifier(num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

        self.save_hyperparameters()
        

    def training_step(self, batch, batch_idx: int = None):
        
        X, y = batch

        output = self.model(X)
        loss = self.loss_fn(output, y)
        self.log("train_loss", loss)

        preds = torch.argmax(output, dim=1)
        acc = (preds == y).float().mean()
        self.log("train_acc", acc)
        
        return loss

    def validation_step(self, batch, batch_idx: int = None):
        X, y = batch
        
        output = self.model(X)
        loss = self.loss_fn(output, y)
        self.log("val_loss", loss)

        preds = torch.argmax(output, dim=1)
        acc = (preds == y).float().mean()
        self.log("val_acc", acc)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, x):
        return self.model(x)

    