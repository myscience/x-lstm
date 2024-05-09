import torch
import torch.nn as nn

from einops import rearrange

from torch import Tensor
from typing import List, TypeVar

T = TypeVar('T')
D = TypeVar('D')

def exists(var : T | None) -> bool:
    return var is not None

def default(var : T | None, val : D) -> T | D:
    return var if exists(var) else val

def enlarge_as(src : Tensor, other : Tensor) -> Tensor:
    '''
        Add sufficient number of singleton dimensions
        to tensor a **to the right** so to match the
        shape of tensor b. NOTE that simple broadcasting
        works in the opposite direction.
    '''
    return rearrange(src, f'... -> ...{" 1" * (other.dim() - src.dim())}').contiguous()

class CausalConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True
    ):
        self._padding = (kernel_size - 1) * dilation

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self._padding,
            dilation=dilation,
            groups=groups,
            bias=bias)

    def forward(self, inp : Tensor) -> Tensor:
        # Handle the case where input has only two dimensions
        # we expect them to have semantics (batch, channels),
        # so we add the missing dimension manually
        if inp.dim() == 2: inp = rearrange(inp, 'b i -> b 1 i')
        
        result = super(CausalConv1d, self).forward(inp)
        if self._padding != 0: return result[..., :-self._padding]
        return result
    
class BlockLinear(nn.Module):
    def __init__(
        self,
        block_dims : List[int | List[int]],
        bias : bool = False,
    ):
        super(BlockLinear, self).__init__()
        
        self._blocks = nn.ParameterList([
            nn.Parameter(torch.randn(size, requires_grad=True))
            for size in block_dims
        ])
        
        self._bias = nn.Parameter(torch.zeros(sum(block_dims))) if bias else None
        
    def forward(self, inp : Tensor) -> Tensor:
        # Assemble the blocks into a block-diagonal matrix
        full = torch.block_diag(*self._blocks)
        
        out = torch.matmul(full, inp)
        
        if self._bias is not None:
            out = out + self._bias
        
        return out