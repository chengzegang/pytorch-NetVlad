# pytorch-NetVlad

This is a model hub for the original [NetVlad](https://arxiv.org/abs/1511.07247) repo, see [here](https://github.com/Nanne/pytorch-NetVlad), used a slightly variant implementation.


# Usage

```python
import torch.hub
model = torch.hub.load('chengzegang/pytorch-NetVlad', 'NETVLAD_VGG16_PITTSBURGH', pretrained=True, optimize=True, optimize_for_inference=False)
```