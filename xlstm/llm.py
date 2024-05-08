import torch.nn as nn

class xLSTM(nn.Module):
    '''The extended Long Short Term Memory (xLSTM) module as
    originally introduced in Beck et al. (2024)] see:
    (https://arxiv.org/abs/2405.04517).
    
    This model stacks sLSTM and mLSTM modules with residual
    connections and offers superior memory and performance
    compared to the standard LSTM model, achieving competitive
    or better performance and scaling than Transformer models
    or State-Space models.
    '''
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)