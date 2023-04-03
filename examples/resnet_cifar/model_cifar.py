# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, List, Optional, Type, Union
import torch
from torch.nn import functional as F
import torch.nn as nn
from torch.nn import Parameter
from composer.loss import binary_cross_entropy_with_logits, soft_cross_entropy
from composer.metrics import CrossEntropy
from composer.models import ComposerClassifier
from torchmetrics import Accuracy, MetricCollection
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
from torchvision.models import resnet
# from torchvision.models.resnet import Bottleneck
# from sample.pytorch.py_arch.bayes.resnet import BayesResNet2

class BayesResNet(ResNet):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__(
            block,
            layers,
            num_classes,
            zero_init_residual,
            groups,
            width_per_group,
            replace_stride_with_dilation,
            norm_layer,
        )

        avgpool1 = nn.AdaptiveAvgPool2d((2, 4))
        avgpool2 = nn.AdaptiveAvgPool2d((2, 2))
        avgpool3 = nn.AdaptiveAvgPool2d((2, 1))
        self.avgpools = nn.ModuleList([
            avgpool1,
            avgpool2, 
            avgpool3,
            self.avgpool,
        ])
        log_prior = torch.zeros(1, num_classes)
        self.register_buffer('log_prior', log_prior)
        self.logits_bias = Parameter(torch.zeros(1, num_classes))

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, _, _ = x.shape
        log_prior = self.log_prior.repeat(batch_size, 1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for i, layer in enumerate([
            self.layer1, self.layer2,
            self.layer3, self.layer4
        ]):
            for block in layer:
                x = block(x)
                logits = self.avgpools[i](x)
                logits = torch.flatten(logits, start_dim=1)
                logits = self.fc(logits)
                log_prior = log_prior + logits
                log_prior = log_prior - torch.mean(log_prior, dim=-1, keepdim=True) + self.logits_bias
                log_prior = F.log_softmax(log_prior, dim=-1)
        return log_prior
    

class BayesResNet2(ResNet):
    def __init__(self, 
                block: Type[Union[BasicBlock, Bottleneck]], 
                layers: List[int], 
                num_classes: int = 1000, 
                zero_init_residual: bool = False, 
                groups: int = 1, 
                idth_per_group: int = 64, 
                replace_stride_with_dilation: Optional[List[bool]] = None, 
                norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super().__init__(block, 
                        layers, 
                        num_classes, 
                        zero_init_residual, 
                        groups, 
                        idth_per_group,
                        replace_stride_with_dilation, 
                        norm_layer)
        # self.fc = nn.Linear(512 * block.expansion, 1000) 
        expansion = block.expansion
        self.digups = nn.ModuleList([
            *[nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(64 * i * expansion, num_classes)
                ) for i in (1, 2, 4) 
                ],
            nn.Sequential(
                self.avgpool,
                nn.Flatten(),
                self.fc,
            )
        ])

        log_prior = torch.zeros(1, num_classes)
        self.register_buffer('log_prior', log_prior)
        self.logits_layer_norm = nn.LayerNorm(num_classes)
        # self.digup = nn.Linear(num_classes, 10)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, _, _ = x.shape
        log_prior = self.log_prior.repeat(batch_size, 1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for i, layer in enumerate([
            self.layer1, self.layer2,
            self.layer3, self.layer4
        ]):
            for block in layer:
                x = block(x)
                logits = self.digups[i](x)
                log_prior = log_prior + logits
                log_prior = self.logits_layer_norm(log_prior)
                # log_prior = log_prior - torch.mean(log_prior, dim=-1, keepdim=True) + self.logits_bias
                # log_prior = F.log_softmax(log_prior, dim=-1)
        return log_prior
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_impl(x)
        # x = self.digup(x)
        return x


def build_composer_resnet(model_name: str = 'resnet50',
                          loss_name: str = 'cross_entropy',
                          num_classes: int = 1000):
    """Helper function to build a Composer ResNet model.

    Args:
        model_name (str, optional): Name of the ResNet model to use, either
            ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']. Default: ``'resnet50'``.
        loss_name (str, optional): Name of the loss function to use, either ['cross_entropy', 'binary_cross_entropy'].
            Default: ``'cross_entropy'``.
        num_classes (int, optional): Number of classes in the classification task. Default: ``1000``.
    """
    if model_name == 'bayes_resnet50':
        model = BayesResNet2(Bottleneck, [3, 4, 6, 3])
        # in_chans = model.fc.in_features
        # model.fc = nn.Linear(in_chans, 10)
    else:
        model_fn = getattr(resnet, model_name)
        model = model_fn(num_classes=num_classes, groups=1, width_per_group=64)

    # Specify model initialization
    def weight_init(w: torch.nn.Module):
        if isinstance(w, torch.nn.Linear) or isinstance(w, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(w.weight)
        if isinstance(w, torch.nn.BatchNorm2d):
            w.weight.data = torch.rand(w.weight.data.shape)
            w.bias.data = torch.zeros_like(w.bias.data)
        # When using binary cross entropy, set the classification layer bias to -log(num_classes)
        # to ensure the initial probabilities are approximately 1 / num_classes
        if loss_name == 'binary_cross_entropy' and isinstance(
                w, torch.nn.Linear):
            w.bias.data = torch.ones(
                w.bias.shape) * -torch.log(torch.tensor(w.bias.shape[0]))
            
    model.apply(weight_init)

    # Performance metrics to log other than training loss
    train_metrics = Accuracy()
    val_metrics = MetricCollection([CrossEntropy(), Accuracy()])

    # Choose loss function: either cross entropy or binary cross entropy
    if loss_name == 'cross_entropy':
        loss_fn = soft_cross_entropy
    elif loss_name == 'binary_cross_entropy':
        loss_fn = binary_cross_entropy_with_logits
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
