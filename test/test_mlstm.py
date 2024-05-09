import unittest

import torch
from xlstm import mLSTM

class TestMLSTM(unittest.TestCase):
    def setUp(self):        
        self.inp_dim = 10
        self.head_dim = 8
        self.head_num = 4
        self.hid_dim = self.head_num * self.head_dim
        
        self.batch_size = 5
        
        # Create an instance of mLSTM
        self.model = mLSTM(self.inp_dim, self.head_num, self.head_dim)
        self.input = torch.randn(self.batch_size, self.inp_dim)
        
        self.hid_0 = self.model.init_hidden(self.batch_size)
    
    def test_forward(self):

        # Forward pass
        output, next_hid = self.model(self.input, self.hid_0)

        # Check if the output shape is correct
        self.assertEqual(output.shape, (self.batch_size, self.inp_dim))

        self.assertEqual(next_hid[0].shape, (self.batch_size, self.head_num, self.head_dim, self.head_dim))
        self.assertEqual(next_hid[1].shape, (self.batch_size, self.head_num, self.head_dim))
        self.assertEqual(next_hid[2].shape, (self.batch_size, self.head_num))

    def test_backward(self):
        criterion = torch.nn.MSELoss()
        
        # Forward pass
        target = torch.randn(self.batch_size, self.inp_dim)
        output, next_hid = self.model(self.input, self.hid_0)

        # Define target tensor
        target = torch.randn(self.batch_size, self.inp_dim)

        # Compute loss & backward pass
        loss = criterion(output, target)
        loss.backward()

        # Check if gradients are computed for all parameters
        for param in self.model.parameters():
            self.assertIsNotNone(param.grad)

if __name__ == '__main__':
    unittest.main()