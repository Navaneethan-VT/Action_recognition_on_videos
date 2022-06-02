"""
Copyright(c) 2019-2022 Deep Safety GmbH.

All rights not expressly granted by the Licensor remain reserved.

This unpublished material is proprietary to Deep Safety GmbH.

Proprietary software is computer software licensed under exclusive legal right
of the copyright holder. The receipt or possession of this source code and /
or related information does not convey or imply any rights to use, reproduce,
disclose or distribute its contents, or to manufacture, use, or sell anything
that it may describe, in whole or in part unless prior written permission is
obtained from Deep Safety GmbH.

The methods and techniques described herein are considered trade secrets and /
or confidential. You shall not disclose such Confidential Information and
shall use it only in accordance with the terms of the license agreement you
entered into with Deep Safety GmbH.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES.
"""


import pathlib
import torch
import torch.nn.functional as F
import yaml
from torch import nn, optim
from torch.utils.data import DataLoader
import torchmetrics
import argparse

# Loading Dataset and Architecture
from dataset import VideoDataset
from temporal_flow import Temporal_conv

# Pytorch Lightning
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping


class Temp(pl.LightningModule):
    def __init__(self, config):
        super(Temp, self).__init__()

        self.config = config
        self.temporal_conv = Temporal_conv(self.config)
        self.fc1_combined = nn.Linear(512, config["num_classes"])
        self.softmax = nn.Softmax()
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()

        self.save_hyperparameters()

    def forward(self, inputs):
        x = inputs
        x = self.temporal_conv(x)

        return x

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, labels)
        train_accuracy = self.train_acc(outputs, labels)

        return {"loss": loss, "train_accuracy": train_accuracy}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_accuracy = torch.stack([x['train_accuracy'] for x in outputs]).mean()

        self.logger.experiment.add_scalars("Loss", {"Train": avg_loss}, self.current_epoch)
        self.logger.experiment.add_scalars("Accuracy", {"Train": avg_accuracy}, self.current_epoch)

    def configure_optimizers(self):
        if self.config["optimizer"] == "SGD":
            optimizer = optim.SGD(self.parameters(),
                                  lr=self.config["lr_rate"],
                                  weight_decay=self.config["weight_decay"])
        elif self.config["optimizer"] == "Adam":
            optimizer = optim.Adam(self.parameters(),
                                   lr=self.config["lr_rate"],
                                   weight_decay=self.config["weight_decay"])
        else:
            print("Require either SGD or adam as a optimizer")

        return optimizer

    def train_dataloader(self):
        train_dataloader = DataLoader(VideoDataset(self.config), batch_size=self.config["batch_size"], shuffle=True,
                                      num_workers=self.config["num_worker"])

        return train_dataloader

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, labels)
        val_accuracy = self.valid_acc(outputs, labels)

        return {"val_loss": loss, "val_accuracy": val_accuracy, "preds": outputs, "target": labels}

    def val_dataloader(self):
        validation_dataloader = DataLoader(VideoDataset(self.config, mode='val'),
                                           batch_size=self.config["batch_size"], num_workers=self.config["num_worker"])

        return validation_dataloader

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_accuracy = torch.stack([x["val_accuracy"] for x in outputs]).mean()

        self.logger.experiment.add_scalars("Loss", {"Val": avg_loss}, self.current_epoch)
        self.logger.experiment.add_scalars("Accuracy", {"Val": avg_accuracy}, self.current_epoch)

        return {"val_loss": avg_loss, "val_accuracy": avg_accuracy}


def parse_arguments():
    """
    Directory of the Yaml file

    :return: Gets input from the user
    """
    parser = argparse.ArgumentParser(description="User input about location of the files")
    parser.add_argument(
        "yaml_file",
        type=str,
        help="local storage directory of your Json file"
    )
    args = parser.parse_args()

    return pathlib.Path(args.yaml_file)


def main():
    yaml_file = parse_arguments()

    with open(yaml_file) as yaml_load:
        config = yaml.full_load(yaml_load)
    # chk_pt = "/home/navaneethan/two_stream_architecture/tb_logs/new_model/version_4/checkpoints/epoch=242-step=67067.ckpt"

    if config["early_stopping"]:
        early_stopping = EarlyStopping('valid_acc', patience=config["patience"], mode='max')

        trainer = Trainer(max_epochs=config["epoch"],
                          fast_dev_run=config["fast_dev_run"],
                          precision=config["precision"],
                          gpus=config["gpu"],
                          callbacks=[early_stopping],
                          enable_model_summary=True,
                          # weights_summary="full",
                          progress_bar_refresh_rate=1,
                          checkpoint_callback=True,
                          logger=TensorBoardLogger("tb_logs", name="new_model"))

    else:
        trainer = Trainer(max_epochs=config["epoch"],
                          fast_dev_run=config["fast_dev_run"],
                          precision=config["precision"],
                          gpus=config["gpu"],
                          enable_model_summary=True,
                          # weights_summary="full",
                          progress_bar_refresh_rate=1,
                          logger=TensorBoardLogger("tb_logs", name="new_model"))

    # model = Temp(config).load_from_checkpoint(chk_pt)
    model = Temp(config)
    trainer.fit(model)


if __name__ == '__main__':
    main()
