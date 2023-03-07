import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce
from sunyata.pytorch.arch.bayes.core import log_bayesian_iteration


pair = lambda x: x if isinstance(x, tuple) else (x, x)


class Affine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, 1, dim))
        self.b = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        return x * self.g + self.b
    

class PreAffinePostLayerScale(nn.Module):
    def __init__(self, 
                 dim, 
                 depth, 
                 fn):
        super().__init__()
        if depth <= 18:
            init_eps = 0.1
        elif depth > 18 and depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6

        scale = torch.zeros(1, 1, dim).fill_(init_eps)
        self.scale = nn.Parameter(scale)
        self.affine = Affine(dim)
        self.fn =fn

    def forward(self, x):
        return self.fn(self.affine(x)) * self.scale + x


class DeepBayesInferResMlp(nn.Module):
    def __init__(self,
                 image_size: int = 32,
                 patch_size: int = 4,
                 hidden_dim: int = 128,
                #  depth: int = 16,
                 expansion_factor: int = 4,
                 num_layers: int = 8,
                 num_classes: int = 10,
                 channels: int = 3,
                 is_bayes: bool = True,
                 is_prior_as_params: bool = False,

                 ):
        super().__init__()
        image_h, image_w = pair(image_size)
        assert (image_h % patch_size) == 0 and (image_w % patch_size) == 0,'image must be divisible by patch size'
        num_patches = (image_h // patch_size) * (image_w // patch_size)
        wrapper = lambda i, fn: PreAffinePostLayerScale(hidden_dim, i+1, fn)

        self.layers = nn.ModuleList([
            nn.Sequential(
            wrapper(i, nn.Conv1d(num_patches, num_patches, 1)),
            wrapper(i, nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * expansion_factor),
            nn.GELU(),
            nn.Linear(hidden_dim * expansion_factor, hidden_dim)
            ))
            ) for i in range(num_layers)
        ])

        if not is_bayes:
            self.layers = nn.ModuleList([nn.Sequential(*self.layers)])  # to onr layer
        
        self.embed = nn.Sequential(
            Rearrange('b c (h, p1) (w, p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear((patch_size ** 2 ) * channels, hidden_dim)
        )

        self.digup = nn.Sequential(
            Affine(hidden_dim),
            Reduce('b n c -> b c', 'mean'),
            nn.Linear(hidden_dim, num_classes)
        )
        self.is_bayes = is_bayes
        log_prior = torch.zeros(1, num_classes)
        if is_prior_as_params:
            self.log_prior = nn.Parameter(log_prior)
        else:
            self.register_buffer('log_prior', log_prior)

    def forward(self, x):
        batch_size, _, _, _ = x.shape
        log_prior = self.log_prior.repeat(batch_size, 1)
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
            logits = self.digup(x)
            log_prior = log_prior + logits
            log_prior = log_prior - torch.mean(log_prior, dim=-1, keepdim=True) + self.logits_bias
            log_prior = F.log_softmax(log_prior, dim=-1)  # log_bayesian_iteration(log_prior, logits)
        return log_prior
