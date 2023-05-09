# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from composer.loss import binary_cross_entropy_with_logits, soft_cross_entropy
from composer.metrics import CrossEntropy
from composer.models import ComposerClassifier
from torchmetrics import Accuracy, MetricCollection

# from sunyata.pytorch.arch.bayes.core import log_bayesian_iteration
class LKA(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return attn * u
class Atten(nn.Sequential):
    def __init__(self, dim: int, drop_rate: float):
        super().__init__(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(), 
            LKA(dim),
            nn.Dropout(drop_rate),
            nn.Conv2d(dim, dim, 1),
            # nn.Dropout(drop_rate),
        )
class FoldBlock(nn.Module):
    "Basic block of folded ResNet"
    def __init__(self, fold_num:int, Unit:nn.Module, *args, **kwargs):  # , hidden_dim: int, kernel_size: int, drop_rate:float=0.
        super(FoldBlock, self).__init__()
        self.fold_num = fold_num
        units = []
        for i in range(max(1, fold_num - 1)):
            units += [Unit(*args, **kwargs)]
        self.units = nn.ModuleList(units)
        
    def forward(self, *xs: torch.Tensor)-> torch.Tensor:
        xs = list(xs)
        if self.fold_num == 1:
            xs[0] = xs[0] + self.units[0](xs[0])
            # xs[0] = xs[0] * self.units[0](xs[0])
            return xs
        for i in range(self.fold_num - 1):
            xs[i+1] = xs[i+1] + self.units[i](xs[i])
        xs.reverse()
        return xs
def nll_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # target = torch.argmax(target, dim=1)
    return F.nll_loss(input, target)
    # return - (input * target).mean()

class Residual(nn.Module):
    def __init__(self, fn:nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        return self.fn(x) + x

class FoldNet(nn.Module):
    def __init__(self, fold_num: int, hidden_dim: int,num_layers: int, patch_size: int, num_classes: int, drop_rate: float=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
                FoldBlock(fold_num, Atten, hidden_dim, drop_rate)
                for _ in range(num_layers)
            ])
        # if block == Atten:
        #     self.layers = nn.ModuleList([
        #         FoldBlock(fold_num, block, hidden_dim, drop_rate)
        #         for _ in range(num_layers)
        #     ])
        
        self.embed = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(hidden_dim, eps=7e-5),
        )

        self.digup = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(hidden_dim, num_classes)
        )
        self.fold_num = fold_num

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        x = self.embed(x)
        xs = [x for _ in range(self.fold_num)]
        for layer in self.layers:
            xs= layer(*xs)
        x = xs[-1]
        x = self.digup(x)
        return x

class FoldNetRepeat2(FoldNet):
    def __init__(self, fold_num: int, hidden_dim: int, num_layers: int, patch_size: int, num_classes: int, drop_rate: float = 0.1):
        super().__init__(fold_num, hidden_dim, num_layers, patch_size, num_classes, drop_rate)

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        x = self.embed(x)
        xs = x.repeat(1, self.fold_num, 1, 1)
        xs = torch.chunk(xs, self.fold_num, dim = 1)
        for layer in self.layers:
            xs = layer(*xs)
        x = xs[-1]
        x = self.digup(x)
        return x
    
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
                    nn.BatchNorm2d(hidden_dim),
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

class BayesConvMixer2(ConvMixer):
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
        # log_prior = - torch.log(torch.ones(1, num_classes) * num_classes)
        self.register_buffer('log_prior', log_prior) 
        # self.log_prior = nn.Parameter(torch.zeros(1, num_classes))
        # self.sqrt_num_classes = sqrt(num_classes)
        self.logits_bias = nn.Parameter(torch.zeros(1, num_classes))
        # self.logits_layer_norm = nn.LayerNorm(num_classes)
        self.digup = None
        num_stages = 4
        self.digups = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(hidden_dim, num_classes),
            )
            for _ in range(num_stages)
        ])
        self.stage_depth = num_layers // num_stages if num_layers % num_stages == 0 else num_layers // num_stages + 1


    def forward(self, x: torch.Tensor):
        batch_size, _, _, _ = x.shape
        log_prior = self.log_prior.repeat(batch_size, 1)

        x = self.embed(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            logits = self.digups[i // self.stage_depth](x) 
            log_prior = log_prior + logits
            # log_prior = self.logits_layer_norm(log_prior)
            log_prior = log_prior - torch.mean(log_prior, dim=-1, keepdim=True) + self.logits_bias
            log_prior = F.log_softmax(log_prior, dim=-1) # log_bayesian_iteration(log_prior, logits)
        
        return log_prior


def build_composer_resnet(
    *,
    model_name: str = 'FoldNet',
    loss_name: str = "nll_loss",
    hidden_dim: int,
    kernel_size: int,
    patch_size: int,
    num_layers: int,
    fold_num: int,
    num_classes: int = 1000,
    drop_rate: float=0.1    
):
    """Helper function to build a Composer ResNet model.

    Args:
        num_classes (int, optional): Number of classes in the classification task. Default: ``1000``.
    """
    if model_name == 'convmixer':
        model = ConvMixer(hidden_dim, kernel_size, patch_size, num_layers, num_classes)
    elif model_name == 'convmixer-bayes-2':
        model = BayesConvMixer2(hidden_dim, kernel_size, patch_size, num_layers, num_classes)
    elif model_name == 'FoldNet':
        model = FoldNet(fold_num, hidden_dim,num_layers, patch_size, num_classes, drop_rate)
    elif model_name == 'FoldNetRepeat2':
        model = FoldNetRepeat2(fold_num, hidden_dim,num_layers, patch_size, num_classes, drop_rate)
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

    # Choose loss function: either cross entropy or binary cross entropy
    if loss_name == 'cross_entropy':
        loss_fn = soft_cross_entropy
    elif loss_name == 'binary_cross_entropy':
        loss_fn = binary_cross_entropy_with_logits
    elif loss_name == 'nll_loss':
        loss_fn = nll_loss
    else:
        raise ValueError(
            f"loss_name='{loss_name}' but must be either ['cross_entropy', 'binary_cross_entropy']"
        )

    # Wrapper function to convert a image classification PyTorch model into a Composer model
    composer_model = ComposerClassifier(model,
                                        train_metrics=train_metrics,
                                        val_metrics=val_metrics,
                                        loss_fn=loss_fn)
    return composer_model
