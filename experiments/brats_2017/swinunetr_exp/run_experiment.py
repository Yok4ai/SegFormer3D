import os
import sys
import random
import time
import torch
import pytorch_lightning as pl
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import Compose, Activations, AsDiscrete
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.timer import Timer
from pytorch_lightning.loggers import WandbLogger

sys.path.append("../../../")

import yaml
import argparse
import numpy as np
from typing import Dict
from termcolor import colored
from accelerate import Accelerator
from losses.losses import build_loss_fn
from optimizers.optimizers import build_optimizer
from optimizers.schedulers import build_scheduler
from train_scripts.trainer_ddp import Segmentation_Trainer
from architectures.build_architecture import build_architecture
from dataloaders.build_dataset import build_dataset, build_dataloader

class SwinUNETRTrainer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Build model
        self.model = build_architecture(config)
        
        # Loss function
        self.loss_function = build_loss_fn(
            loss_type=config["loss_fn"]["loss_type"],
            loss_args=config["loss_fn"]["loss_args"]
        )
        
        # Metrics
        self.dice_metric = DiceMetric(include_background=True, reduction="mean")
        self.dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
        
        # Post-processing transforms
        self.post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        
        # Best metric tracking
        self.best_metric = -1
        
        # Training metrics
        self.avg_train_loss_values = []
        self.train_loss_values = []
        self.train_metric_values = []
        self.train_metric_values_tc = []
        self.train_metric_values_wt = []
        self.train_metric_values_et = []
        
        # Validation metrics
        self.avg_val_loss_values = []
        self.epoch_loss_values = []
        self.metric_values = []
        self.metric_values_tc = []
        self.metric_values_wt = []
        self.metric_values_et = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch["image"], batch["label"]
        
        # Forward pass
        outputs = self(inputs)
        loss = self.loss_function(outputs, labels)
        
        # Log training loss
        self.log("train_loss", loss, prog_bar=True)
        
        # Post-process outputs
        outputs = [self.post_trans(i) for i in decollate_batch(outputs)]
        
        # Compute metrics
        self.dice_metric(y_pred=outputs, y=labels)
        self.dice_metric_batch(y_pred=outputs, y=labels)
        
        # Log metrics
        train_dice = self.dice_metric.aggregate().item()
        self.log("train_mean_dice", train_dice, prog_bar=True)
        
        # Store metrics
        self.train_metric_values.append(train_dice)
        metric_batch = self.dice_metric_batch.aggregate()
        self.train_metric_values_tc.append(metric_batch[0].item())
        self.train_metric_values_wt.append(metric_batch[1].item())
        self.train_metric_values_et.append(metric_batch[2].item())
        
        # Log individual metrics
        self.log("train_tc", metric_batch[0].item(), prog_bar=True)
        self.log("train_wt", metric_batch[1].item(), prog_bar=True)
        self.log("train_et", metric_batch[2].item(), prog_bar=True)
        
        # Reset metrics
        self.dice_metric.reset()
        self.dice_metric_batch.reset()
        
        return loss

    def validation_step(self, batch, batch_idx):
        val_inputs, val_labels = batch["image"], batch["label"]
        
        # Sliding window inference
        val_outputs = sliding_window_inference(
            val_inputs,
            roi_size=self.config["sliding_window_inference"]["roi"],
            sw_batch_size=self.config["sliding_window_inference"]["sw_batch_size"],
            predictor=self.model,
            overlap=self.config["sliding_window_inference"]["overlap"]
        )
        
        # Compute loss
        val_loss = self.loss_function(val_outputs, val_labels)
        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)
        
        # Post-process outputs
        val_outputs = [self.post_trans(i) for i in decollate_batch(val_outputs)]
        
        # Compute metrics
        self.dice_metric(y_pred=val_outputs, y=val_labels)
        self.dice_metric_batch(y_pred=val_outputs, y=val_labels)
        
        # Log validation dice
        val_dice = self.dice_metric.aggregate().item()
        self.log("val_mean_dice", val_dice, prog_bar=True)
        
        return {"val_loss": val_loss}

    def on_validation_epoch_end(self):
        # Store metrics
        val_dice = self.dice_metric.aggregate().item()
        self.metric_values.append(val_dice)
        
        val_loss = self.trainer.logged_metrics["val_loss"].item()
        self.epoch_loss_values.append(val_loss)
        
        # Calculate average validation loss
        avg_val_loss = sum(self.epoch_loss_values) / len(self.epoch_loss_values)
        self.log("avg_val_loss", avg_val_loss, prog_bar=True)
        self.avg_val_loss_values.append(avg_val_loss)
        
        # Store individual dice scores
        metric_batch = self.dice_metric_batch.aggregate()
        self.metric_values_tc.append(metric_batch[0].item())
        self.metric_values_wt.append(metric_batch[1].item())
        self.metric_values_et.append(metric_batch[2].item())
        
        # Log validation metrics
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_mean_dice", val_dice, prog_bar=True)
        self.log("val_tc", metric_batch[0].item(), prog_bar=True)
        self.log("val_wt", metric_batch[1].item(), prog_bar=True)
        self.log("val_et", metric_batch[2].item(), prog_bar=True)
        
        # Save best model
        if val_dice > self.best_metric:
            self.best_metric = val_dice
            self.best_metric_epoch = self.current_epoch
            torch.save(self.model.state_dict(), "best_metric_model.pth")
            self.log("best_metric", self.best_metric)
        
        # Reset metrics
        self.dice_metric.reset()
        self.dice_metric_batch.reset()

    def configure_optimizers(self):
        optimizer = build_optimizer(
            model=self.model,
            optimizer_type=self.config["optimizer"]["optimizer_type"],
            optimizer_args=self.config["optimizer"]["optimizer_args"]
        )
        
        scheduler = build_scheduler(
            optimizer=optimizer,
            scheduler_type="training_scheduler",
            config=self.config
        )
        
        return [optimizer], [scheduler]

def launch_experiment(config_path):
    # Load config
    config = load_config(config_path)
    
    # Set seed
    seed_everything(config)
    
    # Build directories
    build_directories(config)
    
    # Build datasets and dataloaders
    trainset = build_dataset(
        dataset_type=config["dataset_parameters"]["dataset_type"],
        dataset_args=config["dataset_parameters"]["train_dataset_args"],
    )
    trainloader = build_dataloader(
        dataset=trainset,
        dataloader_args=config["dataset_parameters"]["train_dataloader_args"],
        config=config,
        train=True,
    )
    
    valset = build_dataset(
        dataset_type=config["dataset_parameters"]["dataset_type"],
        dataset_args=config["dataset_parameters"]["val_dataset_args"],
    )
    valloader = build_dataloader(
        dataset=valset,
        dataloader_args=config["dataset_parameters"]["val_dataloader_args"],
        config=config,
        train=False,
    )
    
    # Initialize model
    model = SwinUNETRTrainer(config)
    
    # Set up callbacks
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=5,
        verbose=True,
        mode='min'
    )
    
    timer_callback = Timer(duration="00:11:00:00")
    
    # Initialize wandb logger
    wandb_logger = WandbLogger(
        project=config["project"],
        name=config["wandb_parameters"]["name"],
        group=config["wandb_parameters"]["group"],
        entity=config["wandb_parameters"]["entity"]
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config["training_parameters"]["num_epochs"],
        devices=1,
        accelerator="gpu",
        precision='16-mixed',
        gradient_clip_val=config["clip_gradients"]["clip_gradients_value"],
        log_every_n_steps=1,
        callbacks=[early_stop_callback, timer_callback],
        limit_val_batches=5,
        check_val_every_n_epoch=1,
        logger=wandb_logger,
    )
    
    # Train model
    trainer.fit(model, train_dataloaders=trainloader, val_dataloaders=valloader)

def seed_everything(config):
    seed = config["training_parameters"]["seed"]
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_path: str) -> Dict:
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def build_directories(config: Dict) -> None:
    if not os.path.exists(config["training_parameters"]["checkpoint_save_dir"]):
        os.makedirs(config["training_parameters"]["checkpoint_save_dir"])

    if os.listdir(config["training_parameters"]["checkpoint_save_dir"]):
        raise ValueError("checkpoint exists -- preventing file override -- rename file")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SwinUNETR training script")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="path to yaml config file"
    )
    args = parser.parse_args()
    launch_experiment(args.config) 