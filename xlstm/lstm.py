import torch.nn as nn

class sLSTM(nn.Module):
    '''The scalar-Long Short Term Memory (sLSTM) module as
    originally introduced in Beck et al. (2024)] see:
    (https://arxiv.org/abs/2405.04517).
    
    This model is a variant of the standard LSTM model and
    offers two major improvements:
    - Exponential gating with appropriate state normalization
        to avoid overflows induced by the exponential function.
    - A new memory mixing within heads but not across heads.
    '''
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
class mLSTM(nn.Module):
    '''The matrix-Long Short Term Memory (mLSTM) module as
    originally introduced in Beck et al. (2024)] see:
    (https://arxiv.org/abs/2405.04517).
    
    This model is a variant of the standard LSTM model and
    offers superior memory due to its storing values in a
    matrix instead of a scalar. It is fully parallelizable
    and updates internal memory with the covariance rule.
    '''
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)