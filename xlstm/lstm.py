import torch
import torch.nn as nn

from torch import exp
from torch import tanh
from torch import sigmoid

from torch import Tensor
from typing import Tuple
from torch.nn.functional import silu
from torch.nn.functional import gelu

from .utils import BlockLinear
from .utils import CausalConv1d

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
    
    def __init__(
        self,
        seq_dim : int,
        head_dim : int,
        head_num : int,
    ) -> None:
        super().__init__()
        
        self.inp_norm = nn.LayerNorm(seq_dim)
        self.hid_norm = nn.GroupNorm(head_num, head_dim)
        
        self.causal_conv = CausalConv1d(
            seq_dim,
            seq_dim,
            kernel_size=4
        )
        
        self.W_z = nn.Linear(seq_dim, head_num * head_dim)
        self.W_i = nn.Linear(seq_dim, head_num * head_dim)
        self.W_o = nn.Linear(seq_dim, head_num * head_dim)
        self.W_f = nn.Linear(seq_dim, head_num * head_dim)
        
        self.R_z = BlockLinear([head_dim] * head_num)
        self.R_i = BlockLinear([head_dim] * head_num)
        self.R_o = BlockLinear([head_dim] * head_num)
        self.R_f = BlockLinear([head_dim] * head_num)
        
        # NOTE: The factor of two in the output dimension of the up_proj
        # is due to the fact that the output needs to branch into two
        # separate outputs to account for the the gated GeLU connection.
        # See Fig. 9 in the paper.
        proj_dim = int(4/3 * head_num * head_dim)
        self.up_proj   = nn.Linear(head_num * head_dim, 2 * proj_dim)
        self.down_proj = nn.Linear(proj_dim, head_num * head_dim)
        
    def forward(
        self,
        seq: Tensor,
        hid: Tuple[Tensor, Tensor, Tensor, Tensor],    
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor]]:
        '''Forward pass of the sLSTM model.

        Args:
            seq (Tensor): The input sequence tensor of shape (batch_size, sequence_length, input_dim).
            hid (Tuple[Tensor, Tensor, Tensor, Tensor]): The hidden state tuple containing the cell state,
                normalizer state, hidden state, and stabilizer state.

        Returns:
            Tuple[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor]]: The output tensor with the residual
                connection and the newly updated hidden state tuple.
        '''
        
        b, l, d = seq.shape
        
        # Separate the hidden (previous) state into the cell state,
        # the normalizer state, the hidden state, and the stabilizer state.
        c_tm1, n_tm1, h_tm1, m_tm1 = hid
        
        x_t = self.inp_norm(seq)
        
        # Optional causal convolution block for the input
        # and forget gates. See Fig. 9 in the paper.
        x_c = self.causal_conv(x_t)
        x_c = silu(x_c)
        
        # Project the input to the different heads for all
        # the gates.
        # NOTE: For input (i) and forget (f) inputs we use
        # the output of the causal conv. See Fig. 9 in the paper.
        i_t: Tensor = self.W_i(x_c) + self.R_i(h_tm1) 
        f_t: Tensor = self.W_f(x_c) + self.R_f(h_tm1) 
        z_t: Tensor = self.W_z(x_t) + self.R_z(h_tm1)
        o_t: Tensor = self.W_o(x_t) + self.R_o(h_tm1)
        
        # Compute the gated outputs for the newly computed inputs
        m_t = torch.max(f_t + m_tm1, i_t)
        
        i_t = exp(i_t - m_t)         # Eq. (16) in ref. paper | or Eq. (38) in supp. mat.
        f_t = exp(f_t - m_t + m_tm1) # Eq. (17) in ref. paper | or Eq. (39) in supp. mat.
        
        z_t = tanh(z_t)              # Eq. (11) in ref. paper
        o_t = sigmoid(o_t)           # Eq. (14) in ref. paper
        
        # Update the internal states of the model
        c_t = f_t * c_tm1 + i_t * z_t # Eq. (8) in ref. paper
        n_t = f_t * n_tm1 + i_t       # Eq. (9) in ref. paper
        h_t = o_t * (c_t / n_t)       # Eq. (10) in ref. paper
        
        # Compute the output of the LSTM block
        out = self.hid_norm(h_t)
        
        # Perform up-and-down projection of the output with
        # projection factor 4/3. See Fig. (9) in supp. mat.
        out1, out2 = self.up_proj(out).chunk(2)
        
        out = out1 + gelu(out2)
        out = self.down_proj(out)
        
        # Return output with the residual connection and the
        # newly updated hidden state.
        return out + seq, (c_t, n_t, h_t, m_t)
        
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