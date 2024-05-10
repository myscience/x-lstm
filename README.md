# xLSTM in Easy Pytorch

This repo contains the _unofficial_ implementation of `xLSTM` model as introduced in [Beck et al. (2024)](https://arxiv.org/abs/2405.04517). This repo is developed mainly for didactic purposes to spell out the details of a modern `Long-Short Term Memory` with competitive performances against modern `Transformers` or `State-Space` models (e.g. `Mamba`).

Just for fun, this repo tries to implement a basic LLM (see `📂 xlstm.llm`) using [Lightning](https://lightning.ai/docs/pytorch/stable/) so that training on multi-gpu (should) be just one variable away.

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

This repo also provides an implementation of an `xLSTM` LLM (which is simply a stack of `sLSTM`s and `mLSTM` plus a prediction head) built using `Pytorch Lightning` which unlocks easy training on multi-gpus. To use it one can simply run the following example:

```python
from lightning import Trainer
from transformers import AutoTokenizer

from xlstm import xLSTM
from xlstm.stories import TinyStoriesLightning

config = ... # path to YAML configuration file

# Load an off-the-shelf tokenizer from HF
tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2')

# Load the Mamba model from a config file
model = xLSTM.from_config(config, key='llm')

# Load the dataset
dataset = TinyStoriesLightning.from_config(
    config,
    tokenizer,
    key='dataset'
)

trainer = Trainer(
  max_epochs  = 500,
  accelerator = 'gpu',
  devices     = 4, # Piece of cake multi-gpu support!
  strategy    = 'ddp_find_unused_parameters_false',
)

# Train the model
trainer.fit(model, dataset)
```

Alternatively, one can also run the training script `train.py` directly which expects the configuration file path and accepts all the Trainer arguments.

```bash
python train.py --config <path_to_config_file>\
  --max_epochs 500\
  --accelerator gpu\
  --devices 4
```

A cool feature of `xLSTM` current implementation is the lazy (batched-) inference implemented via a generator. One can thus print tokens on screen as they are streamed by the model, no need to wait for the whole inference to finish! A mock-up script would look like the following.

```python
from xlstm import xLSTM
from transformers import AutoTokenizer

# Get an off-the-shelf tokenizer
tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2')

tokenizer.add_special_tokens({'pad_token': '<|pad|>'})

# Parameters for the LLM
vocab_size = tokenizer.vocab_size + 1
num_layers = 8
signature = (7, 1)
inp_dim = 16
head_dim = 8
head_num = 4
ker_size = 4
p_factor = (2, 4/3)

model = xLSTM(
    vocab_size = vocab_size,
    num_layers = self.num_layers,
    signature = self.signature,
    inp_dim= self.inp_dim,
    head_dim= self.head_dim,
    head_num= self.head_num,
    p_factor= self.p_factor,
    ker_size = self.ker_size,
)

# Parameters for the inference
token_lim = 16
use_top_k = 50
temperature = 0.7

# Generate text
stream = model.generate(
  # We can provide more than one prompt!
  prompt=[
      'Once upon a time',
      'In a galaxy far far away',
  ],
  tokenizer=tokenizer,
  token_lim=token_lim,
  use_top_k=use_top_k,
  temperature=temperature,
)

for token in stream:
    # Each token is a dictionary indexed by the
    # batch-id and contains the produced string
    # as value, so we can print the first batch as:
    print(token[0], end='')
```

# Roadmap

- [x] Put all the essential pieces together (i.e. `sLSTM` & `mLSTM`)
- [x] Add implementation for a full `xLSTM`
- [x] Add functioning training script (Lightning)
- [ ] Show some results

# Requirements

To install the required dependencies simply run `pip install -r requirements.txt`.

```
torch==2.3.0
PyYAML==6.0.1
einops==0.8.0
lightning==2.2.4
setuptools==69.5.1
transformers==4.40.2
```

# Citations

```bibtex
@article{beck2024xlstm,
  title={xLSTM: Extended Long Short-Term Memory},
  author={Beck, Maximilian and P{\"o}ppel, Korbinian and Spanring, Markus and Auer, Andreas and Prudnikova, Oleksandra and Kopp, Michael and Klambauer, G{\"u}nter and Brandstetter, Johannes and Hochreiter, Sepp},
  journal={arXiv preprint arXiv:2405.04517},
  year={2024}
}
```
