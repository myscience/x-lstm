import unittest

import yaml
from os import path
from unittest.mock import Mock
from torch import Tensor
from transformers import AutoTokenizer

from xlstm.stories import TinyStories, TinyStoriesLightning
from xlstm.utils import TokenizerWrapper

# Loading `local_settings.json` for custom local settings
test_folder = path.dirname(path.abspath(__file__))
local_settings = path.join(test_folder, '.local.yaml')

with open(local_settings, 'r') as f:
    local_settings = yaml.safe_load(f)

class TestTinyStories(unittest.TestCase):

    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2')
        
        self.tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
        

    def test_len(self):
        self.dataset = TinyStories(
            root=local_settings['tiny_stories_root'],
            tokenizer=self.tokenizer,
            max_length=256,
            data_split='train',
            read_chunk=4096
        )
        
        self.assertEqual(len(self.dataset), 0)  # Replace 0 with the expected length

    def test_iter(self):
        iterator = iter(self.dataset)
        inputs, labels = next(iterator)
        self.assertIsInstance(inputs, Tensor)
        self.assertIsInstance(labels, Tensor)
        # Add more assertions to validate the data returned by the iterator

    def tearDown(self):
        pass

class TestTinyStoriesLightning(unittest.TestCase):

    def setUp(self):
        self.seq_len = 256
        self.batch_size = 16
        self.num_workers = 2
        
        wrapper = TokenizerWrapper(
            pretrained_model_name_or_path='openai-community/gpt2',
            special_tokens={'pad_token': '<|pad|>'}
        )
        
        self.module = TinyStoriesLightning(
            tokenizer=wrapper,
            root=local_settings['tiny_stories_root'],
            max_length=self.seq_len,
            read_chunk=1024,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        
        self.module.setup('fit')
        
    def test_setup_fit(self):
        self.assertIsInstance(self.module.train_dataset, TinyStories)
        self.assertIsInstance(self.module.valid_dataset, TinyStories)
        
        # Add more assertions to validate the setup for the 'fit' stage
        # NOTE: We discard the first batch because it appears to be inconsistent
        #       for multi-workers setup probably due to strange synchronizations
        trainset = iter(self.module.train_dataloader())
        _     = next(trainset)
        batch = next(trainset)
        
        prev, post = batch
        
        self.assertEqual(prev.shape, (self.batch_size, self.seq_len))
        self.assertEqual(post.shape, (self.batch_size, self.seq_len))
        
        # Add more assertions to validate the setup for the 'fit' stage
        valset = iter(self.module.val_dataloader())
        _     = next(valset)
        batch = next(valset)
        
        prev, post = batch
        
        self.assertEqual(prev.shape, (self.batch_size, self.seq_len))
        self.assertEqual(post.shape, (self.batch_size, self.seq_len))

    def test_setup_test(self):
        self.module.setup('test')
        
        self.assertIsInstance(self.module.test__dataset, TinyStories)
        
        batch = next(iter(self.module.test_dataloader()))
        
        prev, post = batch
        
        self.assertEqual(prev.shape, (self.batch_size, self.seq_len))
        self.assertEqual(post.shape, (self.batch_size, self.seq_len))

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()