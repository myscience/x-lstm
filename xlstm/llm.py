import torch
import torch.nn as nn
from lightning import LightningModule

from torch import Tensor
from torch.optim import AdamW, Optimizer
from torch.nn .functional import softmax
from torch.nn.functional import cross_entropy

from typing import Dict, Generator, List, Tuple
from transformers import PreTrainedTokenizerBase

from itertools import repeat
from einops import rearrange


from .lstm import sLSTM
from .lstm import mLSTM
from .utils import Hidden

class xLSTM(LightningModule):
    '''The extended Long Short Term Memory (xLSTM) module as
    originally introduced in Beck et al. (2024)] see:
    (https://arxiv.org/abs/2405.04517).
    
    This model stacks sLSTM and mLSTM modules with residual
    connections and offers superior memory and performance
    compared to the standard LSTM model, achieving competitive
    or better performance and scaling than Transformer models
    or State-Space models.
    '''
    
    @classmethod
    def from_config(cls, config: Dict[str, int]) -> 'xLSTM':
        pass
    
    def __init__(
        self, 
        vocab_size : int,
        num_layers : int,
        signature : Tuple[int, int],
        inp_dim : int,
        head_dim : int,
        head_num : int,
        p_factor : Tuple[float, float] = (2, 4/3),
        ker_size : int = 4,
        tokenizer: PreTrainedTokenizerBase | None = None,
        **kwargs,
    ) -> None:
        '''Initialize the LLM model.

        Args:
            vocab_size (int): The size of the vocabulary.
            num_layers (int): The number of layers in the LLM model.
            signature (Tuple[int, int]): The signature of the LLM model,
                which represents the ration of the mLSTM-to-sLSTM blocks.
            inp_dim (int): The dimension of the input tokens.
            head_dim (int): The dimension of each attention head.
            head_num (int): The number of attention heads.
            p_factor (Tuple[float, float], optional): The expansion factor
                for the MLP projection in the m|s-LSTM blocks. Defaults to (2, 4/3).
            ker_size (int, optional): The kernel size for the causal convolutional layers.
                Defaults to 4.
                
            kwargs: Additional keyword arguments used at inference time (see relevant
                arguments of the generate method).
        '''
        super().__init__()
        
        self.tokenizer = tokenizer
        self.inference_kw = kwargs
        
        # Needed embedding layer for mapping input tokens to the network
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=inp_dim,
        )
        
        m_factor, s_factor = p_factor
        
        mlstm_par = {
            'inp_dim' : inp_dim,
            'head_dim' : head_dim,
            'head_num' : head_num,
            'p_factor' : m_factor,
            'ker_size' : ker_size,
        }
        
        slstm_par = {
            'inp_dim' : inp_dim,
            'head_dim' : head_dim,
            'head_num' : head_num,
            'p_factor' : s_factor,
            'ker_size' : ker_size,
        }
        
        m_num, s_num = signature
        which = [True] * m_num + [False] * s_num
        
        self.llm : List[mLSTM | sLSTM] = nn.ModuleList([
            mLSTM(**mlstm_par) if w else sLSTM(**slstm_par)
            for w, _ in zip(repeat(which), range(num_layers))
        ])
        
        # Prediction head to map the output of the xLSTM model to the vocabulary
        self.head = nn.Linear(inp_dim, vocab_size, bias=False)
        
        self.save_hyperparameters()
        
    def forward(
        self,
        tok: Tensor,
        hid: Hidden | None = None,
        batch_first : bool = False,
    ) -> Tuple[Tensor, Hidden]:
        '''Forward pass of the xLSTM model.

        Args:
            tok (Tensor): Input tensor representing the sequence tokens.
                Expected shape: (batch, seq_len) if batch_first=True,
                else (seq_len, batch).
            hid (Hidden, optional): Cache object for storing intermediate hidden
                values of the m|s-LSTM blocks of the model. If None, the hidden
                states are initialized by the models. Defaults to None.

        Returns:
            Tuple[Tensor, Hidden]: Output tensor after passing through the model
                and updated hidden state.
        '''
        
        tok : Tensor = torch.atleast_2d(tok)
        seq : Tensor = self.embedding(tok)
        
        if batch_first: seq = rearrange(seq, 'b s i -> s b i')
        if hid is None: hid = [l.init_hidden(seq.size(1)) for l in self.llm]
        
        # Pass the sequence through the mLSTM and sLSTM blocks
        out = []
        for inp in seq:
            # Compute model output and update the hidden states
            for i, lstm in enumerate(self.llm):
                inp, hid[i] = lstm(inp, hid[i])
            
            out.append(inp)
            
        out = torch.stack(out, dim=1 if batch_first else 0)
        out = self.head(out)
        
        return out, hid

    @torch.no_grad()
    def generate(
        self,
        prompt : str | List[str],
        tokenizer : PreTrainedTokenizerBase, 
        token_lim : int = 300,
        use_top_k : int = 50,
        temperature : float = 1.0,
    ) -> Generator[Dict[int, str], None, None]:
        # Set model in evaluation model for inference
        self.eval()
        
        if isinstance(prompt, str):
            prompt = [prompt]
        
        # Encode the prompt using the tokenizer
        inp = tokenizer(
            prompt,
            return_tensors='pt',
            padding=True,
            truncation=True,
        ).input_ids
        
        batch_size, inp_len = inp.shape
        vocab_size = tokenizer.vocab_size # type: ignore
        
        # Consume the prompt to get the hidden states
        for tok in rearrange(inp, 'b s -> s b 1'):
            logits, cache = self(tok, cache)
        
        # Start generating the output sequence until either the maximum
        # token limit is reach or the model generates the<|endoftext|> token
        num_tokes = 0
        out, pred = [inp], tok
        pidx = torch.arange(batch_size)
        
        yield {int(pid) : tokenizer.decode(raw, skip_special_tokens=True) for pid, raw in zip(pidx, inp)}
        
        while num_tokes < token_lim and len(pred):
            logits, cache = self(pred, cache)
            
            # Get the token with the highest probability by zeroing out
            # the probability of the lowest probability tokens
            prob = softmax(logits[:, -1] / temperature, dim=-1)
            idxs = prob.topk(k=vocab_size - use_top_k, largest=False, sorted=False).indices
            prob.scatter_(dim=-1, index=idxs, src=torch.zeros_like(prob))
            prob /= prob.sum(dim=-1, keepdim=True)
            
            # Sample the next token from the distribution modelled by the llm
            pred = torch.multinomial(prob, num_samples=1, replacement=True)
            
            # Append the token to the input sequence
            out.append(pred)
            
            num_tokes += 1
            
            # Drop from the batch every prediction that reached the <|endoftext|> token
            mask = pred.squeeze() != tokenizer.eos_token_id
            
            pred  = pred[mask]
            pidx  = pidx[mask]
            cache = (cache[0][mask], cache[1][mask])
            
            # Yield the decoded tokens
            yield {int(pid) : tokenizer.decode(raw, skip_special_tokens=True) for pid, raw in zip(pidx, pred)}
        
        self.train()
    
    def compute_loss(self, prev : Tensor, post : Tensor) -> Tensor:
            '''Compute the cross-entropy loss between the predicted (logits) and
            the actual next token, for all tokens in the batch.

            Args:
                prev (Tensor): The tensor containing the previous tokens.
                    Expected shape: (batch, seq_len).
                post (Tensor): The tensor containing the next tokens, i.e.
                    the targets. Expected shape: (batch, seq_len).

            Returns:
                Tensor: The computed loss between the predicted tokens and the target tokens.
            '''
            # Compute model predictions (logits) for the next tokens based
            # on the previous tokens
            pred, _ = self(prev)

            pred = rearrange(pred, 'b s v -> (b s) v')
            post = rearrange(post, 'b s -> (b s)')
            
            # Compute the loss using the cross entropy loss
            loss = cross_entropy(pred, post)
            
            return loss
    
    def training_step(self, batch : Tuple[Tensor, Tensor], batch_idx : int) -> Tensor:
        prev_tok, next_tok = batch
        
        loss = self.compute_loss(prev_tok, next_tok)

        self.log_dict(
            {'train_loss' : loss},
            logger=True,
            on_step=True,
            sync_dist=True
        )
        
        return loss
    
    def validation_step(self, batch : Tensor, batch_idx : int) -> Tensor:
        prev_tok, next_tok = batch
        
        loss = self.compute_loss(prev_tok, next_tok)

        self.log_dict(
            {'val_loss' : loss},
            logger=True,
            on_step=True,
            sync_dist=True
        )
        
        return loss
    
    def on_validation_end(self) -> None:
        # No need to generate text if the tokenizer is not available
        if self.tokenizer is None: return
        
        inference_kw = {
            'prompt' : 'Once upon a time',
            'tokenizer' : self.tokenizer,
            **self.inference_kw
        }
        
        # Generate the model output on the given prompt
        output = list( # List needed to consume the generator
            self.generate(
                **inference_kw
            )
        )
        
        # Assemble the outputs based on the batch id
        pids = list(output[0].keys())
        output = {pid : ''.join([out[pid] for out in output]) for pid in pids}
        
        for pid, text in output.items():
            self.logger.experiment.add_text({ # type: ignore
                    f'Prompt {pid}' : text
                },
                global_step=self.global_step,
            )
    
    def configure_optimizers(self) -> Optimizer:
        optim = AdamW(
            self.parameters(),
            lr=1e-3
        )
        
        return optim