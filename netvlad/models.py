from typing import Any, Mapping
from .modules import ImageNetVLAD
import torch
from torch import nn, Tensor
import os
import inspect
from abc import ABCMeta
from enum import Enum

TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(TOP_DIR, "checkpoints")
default_compile_backend = "aot_ts_nvfuser"
default_inference_compile_backend = "nvprims_nvfuser"


class PretrainedModel(ABCMeta):
    def __new__(
        cls,
        name,
        bases,
        attrs,
        checkpoint_name: str,
        init_kwargs: Mapping[str, Any],
        **kwargs
    ):
        if name != "PretrainedModel":
            mcls = super().__new__(cls, name, bases, attrs, **kwargs)

            def pretrained_setup(mod: nn.Module):
                mod.load_state_dict(
                    torch.load(os.path.join(CHECKPOINT_DIR, checkpoint_name))
                )
                return mod

            def optimize_(mod: nn.Module, for_inference: bool = False):
                mod = torch.jit.script(mod)
                if for_inference:
                    mod.eval()
                    mod.requires_grad_(False)
                    mod = torch.jit.freeze(mod)
                    mod = torch.jit.optimize_for_inference(mod)
                mod = torch.compile(
                    mod,
                    fullgraph=True,
                    dynamic=False,
                    backend=default_compile_backend
                    if not for_inference
                    else default_inference_compile_backend,
                )
                return mod

            def init_wrapper(
                mod: nn.Module,
                pretrained: bool = True,
                optimize: bool = True,
                optimize_for_inference: bool = False,
                **kwargs
            ):
                default_kwargs = inspect.signature(cls.__init__).parameters.copy()
                default_kwargs.pop("self")
                default_kwargs.pop("args")
                default_kwargs.pop("kwargs")
                default_kwargs.update(init_kwargs)
                default_kwargs.update(kwargs)
                obj = super(mcls, mod).__new__(mcls)
                super(mcls, obj).__init__(**default_kwargs)
                obj = pretrained_setup(obj) if pretrained else obj
                obj = optimize_(obj, optimize_for_inference) if optimize else obj
                return obj

            setattr(mcls, "__new__", init_wrapper)

            return mcls


class NETVLAD_VGG16_PITTSBURGH(
    ImageNetVLAD,
    metaclass=PretrainedModel,
    checkpoint_name="vgg16_netvlad_pittsburger_github.pt",
    init_kwargs={
        "num_clusters": 64,
        "model_dim": 512,
        "output_dim": 512,
        "pretrained": True,
        "output_layer": "none",
    },
):
    ...


class PrerainedModels(Enum):
    NETVLAD_VGG16_PITTSBURGH = NETVLAD_VGG16_PITTSBURGH()
