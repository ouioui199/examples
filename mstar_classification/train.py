# MIT License

# Copyright (c) 2025 Xuan-Huy Nguyen

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Standard imports
from argparse import ArgumentParser

# External imports
import torch

from torchcvnn.datasets import MSTARTargets, SAMPLE

from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

# Local imports
from model import ViTMSTARModule, ViTSAMPLEModule
from utils import CustomProgressBar, TBLogger, PadIfNeeded, ApplyFFT2, ApplyIFFT2, Compose, train_parser, ToTensor, get_dataloaders

        
def lightning_train_MSTAR(opt: ArgumentParser, trainer: Trainer):
    # Dataloading
    train_dataset = MSTARTargets(
        opt.datadir,
        transform=Compose([
            ApplyFFT2(),
            PadIfNeeded(opt.input_size, opt.input_size),
            ApplyIFFT2(),
            ToTensor()
            # TODO: LogTransform
        ])
    )
    valid_dataset = MSTARTargets(
        opt.datadir,
        transform=Compose([
            ApplyFFT2(),
            PadIfNeeded(opt.input_size, opt.input_size),
            ApplyIFFT2(),
            ToTensor()
        ])
    )
    train_loader, valid_loader = get_dataloaders(opt, train_dataset, valid_dataset)
    model = ViTMSTARModule(opt, num_classes=len(train_dataset.class_names))
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)


def lightning_train_SAMPLE(opt: ArgumentParser, trainer: Trainer):
    # Dataloading
    train_dataset = SAMPLE(
        opt.datadir,
        transform=Compose([
            ToTensor()
            # TODO: LogTransform
        ])
    )
    valid_dataset = SAMPLE(
        opt.datadir,
        transform=Compose([
            ToTensor()
        ])
    )
    train_loader, valid_loader = get_dataloaders(opt, train_dataset, valid_dataset)
    model = ViTSAMPLEModule(opt, num_classes=len(train_dataset.class_names))
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser = train_parser(parser)
    opt = parser.parse_args()

    trainer = Trainer(
        max_epochs=opt.epochs,
        num_sanity_val_steps=0,
        benchmark=True,
        enable_checkpointing=True,
        callbacks=[
            CustomProgressBar(),
            EarlyStopping(
                monitor='val_loss', 
                verbose=True,
                patience=opt.patience,
                min_delta=0.005
            ),
            LearningRateMonitor(logging_interval='epoch'),
            ModelCheckpoint(
                dirpath=opt.weightdir,
                monitor='val_Accuracy', 
                verbose=True, 
                mode='max'
            )
        ],
        logger=[
            TBLogger(opt.logdir, name=None, sub_dir='train', version=opt.version),
            TBLogger(opt.logdir, name=None, sub_dir='valid', version=opt.version)
        ]
    )
    
    torch.set_float32_matmul_precision('high')
    lightning_train_MSTAR(opt, trainer)