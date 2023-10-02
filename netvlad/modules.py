from functools import partial
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
import torchvision.models as models  # type: ignore
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from torchvision.models.feature_extraction import create_feature_extractor


class NetVLAD(nn.Module):
    def __init__(
        self,
        num_clusters: int = 64,
        model_dim: int = 512,
        l2_norm: bool = False,
        bias: bool = False,
        eps: float = 1e-8,
    ):
        """
        NetVLAD module that computes the VLAD encoding of an input tensor.

        Args:
            num_clusters (int): Number of clusters to use for vector quantization.
            model_dim (int): Dimension of the input tensor.
            l2_norm (bool): Whether to apply L2 normalization to the output.
            bias (bool): Whether to include a bias term in the convolutional layer.
            eps (float): A small value to add to the softmax denominator for numerical stability.

        Attributes:
            num_clusters (int): Number of clusters to use for vector quantization.
            model_dim (int): Dimension of the input tensor.
            l2_norm (bool): Whether to apply L2 normalization to the output.
            eps (float): A small value to add to the softmax denominator for numerical stability.
            conv (nn.Conv2d): Convolutional layer used for vector quantization.
            centroids (nn.Parameter): Learnable parameter representing the cluster centroids.
        """
        super().__init__()
        self.num_clusters = num_clusters
        self.model_dim = model_dim
        self.l2_norm = l2_norm
        self.eps = eps
        self.conv = nn.Conv2d(model_dim, num_clusters, kernel_size=1, bias=bias)
        self.centroids = nn.Parameter(
            torch.nn.init.kaiming_normal_(torch.randn(num_clusters, model_dim))
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the NetVLAD module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim, height, width).

        Returns:
            torch.Tensor: VLAD encoding of the input tensor, flattened to shape (batch_size, num_clusters * dim).
        """
        # Compute the soft assignments of the input features to the cluster centroids
        w = self.conv(x).flatten(2).transpose(-1, -2)
        w = F.softmax(w + self.eps, dim=-1)

        # Compute the residuals between the input features and the cluster centroids
        d = x.flatten(2).transpose(-1, -2)
        cent = self.centroids.unsqueeze(0).repeat(d.shape[0], 1, 1)
        r = (d.unsqueeze(-2) - cent.unsqueeze(-3)) * w.unsqueeze(-1)

        # Compute the VLAD encoding by summing the residuals and flattening the result
        vlad = r.sum(dim=1)
        if self.l2_norm:
            vlad = F.normalize(vlad, p=2.0, dim=2)
            vlad = F.normalize(vlad.view(vlad.shape[0], -1), p=2.0, dim=1).view_as(vlad)
        vlad = vlad.flatten(1)
        return vlad


class ImageNetVLAD(nn.Module):
    __name__ = "ImageNetVLAD"
    """
    ImageNetVLAD module that extracts features from an image using a VGG16 encoder and NetVLAD pooling.

    Args:
        num_clusters (int): Number of clusters for NetVLAD pooling.
        model_dim (int): Dimension of the input feature map.
        output_dim (int): Dimension of the output feature map.
        pretrained (bool): Whether to use pretrained weights for the VGG16 encoder.
        output_layer (Literal["linear-prob-norm", "linear-prob", "none"] | nn.Module | None):
            Output layer of the module. Can be a string indicating the type of output layer,
            a custom nn.Module, or None.

    Attributes:
        num_clusters (int): Number of clusters for NetVLAD pooling.
        model_dim (int): Dimension of the input feature map.
        output_dim (int): Dimension of the output feature map.
        pretrained (bool): Whether to use pretrained weights for the VGG16 encoder.
        encoder (nn.Sequential): VGG16 encoder.
        pool (NetVLAD): NetVLAD pooling layer.
        out (nn.Module): Output layer of the module.

    Methods:
        forward(x: Tensor) -> Tensor: Forward pass of the module.
        grad_checkpoint() -> nn.Module: Returns a checkpointed version of the module for gradient checkpointing.
    """

    def __init__(
        self,
        num_clusters=64,
        model_dim=512,
        output_dim: int = 512,
        pretrained: bool = False,
        output_layer: Literal["linear-prob-norm", "linear-prob", "none"]
        | nn.Module
        | None = "linear-probe-norm",
    ):
        super().__init__()
        self.num_clusters = num_clusters
        self.model_dim = model_dim
        self.output_dim = output_dim
        self.pretrained = pretrained
        self.encoder = models.vgg16(
            weights=models.VGG16_Weights.IMAGENET1K_FEATURES if pretrained else None
        )
        self.encoder = nn.Sequential(
            *list(self.encoder.features.children())[:-2],
        )
        self.encoder.zero_grad()
        self.pool = NetVLAD(num_clusters, model_dim)
        if isinstance(output_layer, nn.Module):
            self.out = output_layer
        elif output_layer == "linear-probe-norm":
            self.out = nn.Sequential(
                nn.Linear(output_dim, output_dim),
                nn.InstanceNorm1d(output_dim, momentum=0.01, eps=1e-4),
            )
        elif output_layer == "linear-probe":
            self.out = nn.Linear(output_dim, output_dim)
        elif output_layer == "none" or output_layer is None:
            pass

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the ImageNetVLAD module.

        Args:
            x (Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            Tensor: Output tensor of shape (batch_size, output_dim).
        """
        x = self.encoder(x)
        x = self.pool(x)
        if hasattr(self, "out"):
            x = self.out(x)
        return x

    def grad_checkpoint(self) -> nn.Module:
        """
        Returns a checkpointed version of the ImageNetVLAD module for gradient checkpointing.

        Returns:
            nn.Module: Checkpointed version of the module.
        """
        return apply_grad_checkpoints(self)


def apply_grad_checkpoints(module: nn.Module) -> nn.Module:
    """
    Applies gradient checkpoints to a PyTorch module.

    Args:
        module (nn.Module): The PyTorch module to apply gradient checkpoints to.

    Returns:
        nn.Module: The PyTorch module with gradient checkpoints applied.
    """
    from functools import wraps

    def _apply_g_ckpt(mod: nn.Module) -> nn.Module:
        if isinstance(mod, nn.ReLU):
            mod.inplace = False
            mod._org_forward = mod.forward
            mod.forward = wraps(mod._org_forward)(
                partial(checkpoint, mod._org_forward, use_reentrant=False)
            )
        return mod

    module.apply(_apply_g_ckpt)
    return module
