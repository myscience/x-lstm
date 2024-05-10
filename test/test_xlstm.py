import unittest

import yaml
import torch
from os import path

from transformers import AutoTokenizer

from xlstm import xLSTM
from xlstm.stories import TinyStoriesLightning
from xlstm.utils import default_iterdata_worker_init

# Loading `local_settings.json` for custom local settings
test_folder = path.dirname(path.abspath(__file__))
local_settings = path.join(test_folder, '.local.yaml')

class TestXLSTM(unittest.TestCase):
    def setUp(self):
        self.num_layers = 8
        self.signature = (7, 1)
        self.inp_dim = 16
        self.head_dim = 8
        self.head_num = 4
        self.ker_size = 4
        self.p_factor = (2, 4/3)

        self.seq_len = 32
        self.batch_size = 4
        self.vocab_size = 24
        
        # Mockup input for example purposes
        self.seq = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))

    def test_llm_forward(self):
        
        xlstm = xLSTM(
            vocab_size = self.vocab_size,
            num_layers = self.num_layers,
            signature = self.signature,
            inp_dim= self.inp_dim,
            head_dim= self.head_dim,
            head_num= self.head_num,
            p_factor= self.p_factor,
            ker_size = self.ker_size,
        )
        
        
        # Compute the output using the xLSTM architecture
        out, _ = xlstm.forward(self.seq, batch_first=True)
        
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.vocab_size))
            
    def test_llm_dataloader(self):
        
        # Get the local path to tiny stories
        with open(local_settings, 'r') as f:
            root = yaml.safe_load(f)['tiny_stories_path']
            
        # Get an off-the-shelf tokenizer
        tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2')
        
        vocab_size = tokenizer.vocab_size
        
        xlstm = xLSTM(
            vocab_size = vocab_size,
            num_layers = self.num_layers,
            signature = self.signature,
            inp_dim= self.inp_dim,
            head_dim= self.head_dim,
            head_num= self.head_num,
            p_factor= self.p_factor,
            ker_size = self.ker_size,
        )
        
        loader = TinyStoriesLightning(
            root,
            tokenizer,
            max_length=self.seq_len,
            batch_size=self.batch_size,
            worker_init_fn=default_iterdata_worker_init,
        )
        
        loader.setup(stage='fit')
        batch = next(iter(loader.train_dataloader()))
        
        prev, post = batch
        
        logits, _ = xlstm(prev)
        
        loss = xlstm.compute_loss(prev, post)
        
        self.assertTrue((loss >= 0).all())
        self.assertEqual(logits.shape, (*prev.shape, vocab_size))
        
    def test_llm_generate(self):
        # Get an off-the-shelf tokenizer
        tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2')
        
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
        
        vocab_size = tokenizer.vocab_size + 1
        token_lim = 16
        
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
        
        # Generate text
        gen = model.generate(
            prompt=[
                'Once upon a time',
                'In a galaxy far far away',
            ],
            tokenizer=tokenizer,
            token_lim=token_lim,
        )
        
        for tok in gen:
            print(tok[0], end='')
        
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()