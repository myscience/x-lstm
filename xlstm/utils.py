import torch
import torch.nn as nn

from torch import Tensor
from typing import List

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
        self.__padding = (kernel_size - 1) * dilation

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias)

    def forward(self, input : Tensor) -> Tensor:
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0: return result[:, :, :-self.__padding]
        return result
    
class BlockLinear(nn.Module):
    def __init__(
        self,
        block_dims : List[int],
    ):
        super(BlockLinear, self).__init__()
        
        self._blocks = nn.ParameterList([
            nn.Parameter(torch.randn(size, size, requires_grad=True))
            for size in block_dims
        ])
        
    def forward(self, inp : Tensor) -> Tensor:
        # Assemble the blocks into a block-diagonal matrix
        full = torch.block_diag(*self._blocks)
        
        return torch.matmul(full, inp)