import torch
from os import path
from torch import Tensor
from torch.utils.data import IterableDataset

from typing import Literal, Tuple, Generator
from transformers import PreTrainedTokenizerBase

from .data import LightningDataset
from .utils import TokenizerWrapper

class TinyStories(IterableDataset):
    
    def __init__(
        self,
        root : str,
        tokenizer : PreTrainedTokenizerBase,
        max_length : int = 256,
        data_split : Literal['train', 'valid', 'test'] = 'train',
        read_chunk : int = 4096,
    ) -> None:
        super().__init__()
        
        text_path = path.join(root, f'{data_split}.txt')
        
        with open(text_path, 'r', encoding='utf-8') as f:
            # Move the file pointer to the end of the file
            f.seek(0, 2)
            
            # Get the current position of the file pointer, which is the file size
            self.file_size = f.tell()
        
        self.tokenizer = tokenizer
        self.read_chunk = read_chunk
        self.max_length = max_length + 1
        
        self.tokens = []
        self.stream = open(text_path, 'r', encoding='utf-8')
        
        self._start = 0
        self._end   = len(self)

    def __len__(self) -> int:
        return self.file_size
    
    def __del__(self) -> None:
        if hasattr(self, 'stream'): self.stream.close()

    def __iter__(self) -> Generator[Tuple[Tensor, Tensor], None, None]:
        
        self.stream.seek(self._start)
        
        while self.stream.tell() < self._end:
            while len(self.tokens) < self.max_length:
                self.tokens.extend(
                    self.tokenizer.encode(
                        self.stream.read(self.read_chunk)
                    )
                )
            
            tokens, self.tokens = self.tokens[:self.max_length], self.tokens[self.max_length:]
            
            prev = torch.tensor(tokens[:-1])
            post = torch.tensor(tokens[+1:])
            
            yield prev, post

class TinyStoriesLightning(LightningDataset):
    '''Lightning Dataset class for the Tiny Stories dataset. The Tiny
    Stories dataset is a small dataset of short stories, each consisting
    of a few sentences. The dataset is used for training a language model.
    '''
    
    def __init__(
        self,
        tokenizer : TokenizerWrapper,
        root : str = './',
        max_length : int = 256,
        read_chunk : int = 1024,
        **kwargs,    
    ) -> None:
        super().__init__(**kwargs)
        
        self.root = root
        self.tokenizer = tokenizer.get_tokenizer()
        self.max_length = max_length
        self.read_chunk = read_chunk
        
        # NOTE: We ignore the tokenizer key to avoid having
        #       a repetition with the LightningModule
        self.save_hyperparameters(ignore=['tokenizer'])
        
    def setup(self, stage: str) -> None:

        match stage:
            case 'fit':
                self.train_dataset = TinyStories(
                    root=self.root,
                    tokenizer=self.tokenizer,
                    max_length=self.max_length,
                    data_split='train',
                    read_chunk=self.read_chunk 
                )
                self.valid_dataset = TinyStories(
                    root=self.root,
                    tokenizer=self.tokenizer,
                    max_length=self.max_length,
                    data_split='valid',
                    read_chunk=self.read_chunk 
                )
            case 'test':
                self.test__dataset = TinyStories(
                    root=self.root,
                    tokenizer=self.tokenizer,
                    max_length=self.max_length,
                    data_split='test',
                    read_chunk=self.read_chunk
                )
            case _:
                raise ValueError(f'Invalid stage: {stage}')