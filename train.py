import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dataset_utils import PromptTrainDataset
from net.model import PromptIR
from utils.schedulers import LinearWarmupCosineAnnealingLR
from options import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_msssim import ssim


class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)
        self.l1_loss = nn.L1Loss()
        self.eps = 1e-6

    def forward(self, x):
        return self.net(x)

    def ssim_loss(self, pred, target):
        return 1 - ssim(pred, target, data_range=1.0, size_average=True)

    def tv_loss(self, x):
        dh = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]).mean()
        dw = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]).mean()
        return dh + dw

    def charbonnier_loss(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps))

    def training_step(self, batch, batch_idx):
        ([_, _], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        loss_recon = self.l1_loss(restored, clean_patch)
        loss_ssim = self.ssim_loss(restored, clean_patch)
        loss_tv = self.tv_loss(restored)
        loss_charb = self.charbonnier_loss(restored, clean_patch)

        loss = (
            1.0 * loss_recon
            + 0.5 * loss_ssim
            + 0.1 * loss_tv
            + 0.1 * loss_charb
        )

        self.log_dict(
            {
                "train/recon": loss_recon,
                "train/ssim": loss_ssim,
                "train/tv": loss_tv,
                "train/charb": loss_charb,
                "train/total": loss,
            },
            prog_bar=True,
        )

        return loss

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(self.current_epoch)
        # lr = scheduler.get_lr()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer, warmup_epochs=25, max_epochs=250
        )

        return [optimizer], [scheduler]


def main():
    print("Options")
    print(opt)
    if opt.wblogger is not None:
        logger = WandbLogger(project=opt.wblogger, name="PromptIR-Train")
    else:
        logger = TensorBoardLogger(save_dir="logs/")

    trainset = PromptTrainDataset(opt)
    checkpoint_callback = ModelCheckpoint(
        dirpath=opt.ckpt_dir, every_n_epochs=10, save_top_k=-1
    )
    trainloader = DataLoader(
        trainset,
        batch_size=opt.batch_size,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        num_workers=opt.num_workers,
    )

    model = PromptIRModel()

    trainer = pl.Trainer(
        max_epochs=opt.epochs,
        accelerator="gpu",
        devices=opt.num_gpus,
        logger=logger,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model=model, train_dataloaders=trainloader)


if __name__ == "__main__":
    main()
