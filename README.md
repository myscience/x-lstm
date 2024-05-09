# xLSTM in Easy Pytorch

This repo contains the _unofficial_ implementation of `xLSTM` model as introduced in [Beck et al. (2024)](https://arxiv.org/abs/2405.04517). This repo is developed mainly for didactic purposes to spell out the details of a `Long-Short Term Memory` with competitive performances against modern `Transformers` or `State-Space` models (e.g. `Mamba`).

# Usage

The `xlstm` module exposes both the `sLSTM` (scalar-LSTM) and the `mLSTM` (matrix-LSTM) modules. Both expect their input to have shape `(batch_size, d_input)` as they consume an input sequence sequentially. They output the output tensor of same shape as input (the prediction for the next sequence token) plus their updated hidden states (a tuple of tensors).

```python
from xlstm import sLSTM
from itertools import pairwise

seq_len = 32
batch_size = 4

inp_dim = 16
head_dim = 8
head_num = 4

# Create a mock up input sequence
seq = torch.randn(seq_len, batch_size, inp_dim)

lstm = sLSTM(
    inp_dim,        # Input sequence dimension
    head_dim,       # Dimension of each head
    head_num,       # Number of heads
    p_factor=4/3,   # Tunable expansion factor
)

# Initialize the hidden states
hid = lstm.init_hidden(batch_size)

criterion = ... # Pick some loss function, i.e. MSE

# Iterate through the sequence length
loss = 0
for prev, succ in pairwise(seq):
    pred, hid = lstm(prev, hid)

    # Target is the next sequence token
    loss += criterion(pred, succ)

# Compute gradients
loss.backward()
```

# Roadmap

- [x] Put all the essential pieces together (i.e. `sLSTM` & `mLSTM`)
- [ ] Add implementation for a full `xLSTM`
- [ ] Add functioning training script (Lightning)
- [ ] Show some results

# Requirements

To install the required dependencies simply run `pip install -r requirements.txt`.

```
einops==0.8.0
setuptools==69.5.1
torch==2.3.0
```

# Citations

```bibtex

```
