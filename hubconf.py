dependencies = ["torch"]
from netvlad.models import PrerainedModels


def NETVLAD_VGG16_PITTSBURGH(
    pretrained: bool = True,
    optimize: bool = True,
    optimize_for_inference: bool = False,
    **kwargs
):
    return PrerainedModels["NETVLAD_VGG16_PITTSBURGH"].value(
        pretrained, optimize, optimize_for_inference, **kwargs
    )
