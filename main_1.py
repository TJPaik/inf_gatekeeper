#%%
import os
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from fastai.vision.models.xresnet import xresnet18_deeper
from dataloaders.datamodule_1 import DataModule
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

class Model(pl.LightningModule):
    def __init__(self):
        super(Model, self).__init__()
        model = xresnet18_deeper
        self.model1 = model(False, c_in=3, n_out=2, ks=41, stride=13)
        self.model2 = model(False, c_in=3, n_out=2, ks=41, stride=13)
        self.model3 = model(False, c_in=3, n_out=2, ks=41, stride=13)
        self.model4 = model(False, c_in=3, n_out=2, ks=41, stride=13)
        self.model5 = model(False, c_in=1, n_out=2, ks=41, stride=13)
        with torch.no_grad():
            self.model5.eval()
            testing = self.model5(torch.zeros(1, 1, 1000))
            self.model5.train()
        self.n_c = testing.shape[1]
        self.last_layer = nn.Linear(self.n_c * 5, 2)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.step(x)

    def step(self, batch):
        x_w, x1, x2, x3, x4 = batch[0], batch[1][:, :3], batch[1][:, 3:6], batch[1][:, 6:9], batch[1][:, 9:]
        x1 = self.model1(x1)
        x2 = self.model2(x2)
        x3 = self.model3(x3)
        x4 = self.model4(x4)
        x_w = self.model5(x_w)
        return self.last_layer(torch.cat([x1, x2, x3, x4, x_w], 1))

    def training_step(self, batch, batch_idx):
        y_pred = self.step(batch)
        loss = self.loss_fn(y_pred, batch[2])
        self.log('train_loss', loss)
        return loss

    def training_epoch_end(self, outputs):
        sch = self.lr_schedulers()
        self.log('lr', sch.get_last_lr()[0])

    def validation_step(self, batch, batch_idx):
        y_pred = self.step(batch)
        loss = self.loss_fn(y_pred, batch[2])
        return {'val loss': loss.cpu().detach(), 'y': batch[2].detach().cpu(),
                'y_pred': F.softmax(y_pred.detach().cpu(), 1)[:, 1]}

    def validation_epoch_end(self, outputs: list):
        y = torch.cat([el['y'] for el in outputs]).numpy()
        if len(set(y)) == 1:
            return super(Model, self).validation_epoch_end(outputs)
        y_pred = torch.cat([el['y_pred'] for el in outputs]).numpy()
        loss = np.mean([el['val loss'].item() for el in outputs])
        auc = roc_auc_score(y, y_pred)
        acc = accuracy_score(y, y_pred > 0.5)
        npv_10 = self.npv(y, y_pred, 0.10)
        npv_25 = self.npv(y, y_pred, 0.25)
        npv_50 = self.npv(y, y_pred, 0.5)
        npv_75 = self.npv(y, y_pred, 0.75)
        npv_90 = self.npv(y, y_pred, 0.90)
        self.log('val_auc', auc)
        self.log('val',
                 {'auc': auc, 'npv10': npv_10, 'npv25': npv_25, 'npv50': npv_50, 'npv75': npv_75, 'npv90': npv_90,
                  'acc': acc, 'loss': loss})
        return super(Model, self).validation_epoch_end(outputs)

    def test_step(self, batch, batch_idx):
        y_pred = self.step(batch)
        loss = self.loss_fn(y_pred, batch[2])
        return {'test loss': loss.cpu().detach(), 'y': batch[2].detach().cpu(),
                'y_pred': F.softmax(y_pred.detach().cpu(), 1)[:, 1]}

    def test_epoch_end(self, outputs) -> None:
        y = torch.cat([el['y'] for el in outputs]).numpy()
        y_pred = torch.cat([el['y_pred'] for el in outputs]).numpy()
        loss = np.mean([el['test loss'].item() for el in outputs])
        # auc = roc_auc_score(y, y_pred)
        acc = accuracy_score(y, y_pred > 0.5)
        npv_10 = self.npv(y, y_pred, 0.10)
        npv_25 = self.npv(y, y_pred, 0.25)
        npv_50 = self.npv(y, y_pred, 0.5)
        npv_75 = self.npv(y, y_pred, 0.75)
        npv_90 = self.npv(y, y_pred, 0.90)
        # self.log('test_auc', auc)
        self.log('test',
                 {'npv10': npv_10, 'npv25': npv_25, 'npv50': npv_50, 'npv75': npv_75, 'npv90': npv_90,
                  'acc': acc, 'loss': loss})
        return super(Model, self).validation_epoch_end(outputs)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.01)
        sch = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=50, max_epochs=500)
        return [optimizer], [sch]

    @staticmethod
    def npv(y_true, y_pred, threshold):
        tmp = confusion_matrix(y_true, [0 if el < threshold else 1 for el in y_pred])[:, 0]
        if sum(tmp) == 0:
            return -1.0
        else:
            return tmp[0] / sum(tmp)


model = Model()
data_module = DataModule(1, 256, 24)

trainer = pl.Trainer(max_epochs=500, gpus=1, log_every_n_steps=1)
trainer.fit(model, datamodule=data_module)

history = trainer.test(model, datamodule=data_module)
