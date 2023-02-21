# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F
from composer.metrics import CrossEntropy
from composer.models import ComposerClassifier
from torchmetrics import Accuracy, MetricCollection

from sunyata.pytorch.arch.base import Residual
from sunyata.pytorch.arch.bayes.core import log_bayesian_iteration


def nll_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = torch.argmax(target, dim=1)
    return F.nll_loss(input, target)
    # return - (input * target).mean()


class ConvMixer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        kernel_size: int,
        patch_size: int,
        num_layers: int,
        num_classes: int,
    ):
        super().__init__()

        self.layers = nn.Sequential(*[
            nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size, groups=hidden_dim, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(hidden_dim)
                )),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(hidden_dim)
            ) for _ in range(num_layers)
        ])

        self.embed = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(hidden_dim),
        )

        self.digup = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        x= self.layers(x)
        x= self.digup(x)
        return x


class BayesConvMixer(ConvMixer):
    def __init__(
        self,
        hidden_dim: int,
        kernel_size: int,
        patch_size: int,
        num_layers: int,
        num_classes: int,
    ):
        super().__init__(hidden_dim, kernel_size, patch_size, num_layers, num_classes)

        log_prior = torch.zeros(1, num_classes)
        self.register_buffer('log_prior', log_prior) 

    def forward(self, x):
        batch_size, _, _, _ = x.shape
        log_prior = self.log_prior.repeat(batch_size, 1)

        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
            logits = self.digup(x) 
            log_prior = log_bayesian_iteration(log_prior, logits)
        
        return log_prior


def build_composer_resnet(
    *,
    model_name: str = 'convmixer',
    hidden_dim: int,
    kernel_size: int,
    patch_size: int,
    num_layers: int,
    num_classes: int = 1000    
):
    """Helper function to build a Composer ResNet model.

    Args:
        num_classes (int, optional): Number of classes in the classification task. Default: ``1000``.
    """
    if model_name == 'convmixer':
        model = ConvMixer(hidden_dim, kernel_size, patch_size, num_layers, num_classes)
    elif model_name == 'convmixer-bayes':
        model = BayesConvMixer(hidden_dim, kernel_size, patch_size, num_layers, num_classes)
    else:
        raise ValueError("Only support convmixer and convmixer-bayes till now.")

    # Specify model initialization
    def weight_init(w: torch.nn.Module):
        if isinstance(w, torch.nn.Linear) or isinstance(w, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(w.weight)
        if isinstance(w, torch.nn.BatchNorm2d):
            w.weight.data = torch.rand(w.weight.data.shape)
            w.bias.data = torch.zeros_like(w.bias.data)

    model.apply(weight_init)

    # Performance metrics to log other than training loss
    train_metrics = Accuracy()
    val_metrics = MetricCollection([CrossEntropy(), Accuracy()])

    # Wrapper function to convert a image classification PyTorch model into a Composer model
    composer_model = ComposerClassifier(model,
                                        train_metrics=train_metrics,
                                        val_metrics=val_metrics,
                                        loss_fn=nll_loss)
    return composer_model
