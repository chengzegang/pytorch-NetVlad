dependencies = ["torch"]
from netvlad.models import PrerainedModels


def IMAGENETVLAD_VGG16_PITTSBURGH_GITHUB(
    pretrained: bool = True,
    optimize: bool = True,
    optimize_for_inference: bool = False,
    **kwargs
):
    return PrerainedModels["IMAGENETVLAD_VGG16_PITTSBURGH_GITHUB"].value(
        pretrained, optimize, optimize_for_inference, **kwargs
    )
