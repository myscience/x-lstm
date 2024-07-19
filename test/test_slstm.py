import unittest
import torch
from xlstm import sLSTM

class TestSLSTM(unittest.TestCase):
    def setUp(self):        
        self.inp_dim = 10
        self.head_dim = 8
        self.head_num = 4
        self.hid_dim = self.head_num * self.head_dim
        
        self.batch_size = 5
        
        self.model = sLSTM(self.inp_dim, self.head_dim, self.head_num)
        self.input = torch.randn(self.batch_size, self.inp_dim)
        
        self.hid_0 = self.model.init_hidden(self.batch_size)

    def test_output_shape(self):
        output, _ = self.model(self.input, self.hid_0)
        
        self.assertEqual(output.shape, (self.batch_size, self.inp_dim))

    def test_hidden_shape(self):
        hid = self.model.init_hidden(self.batch_size)
        self.assertEqual(len(hid), 4) 
        
        self.assertEqual(hid[0].shape, (self.batch_size, self.hid_dim,))
        self.assertEqual(hid[1].shape, (self.batch_size, self.hid_dim,))
        self.assertEqual(hid[2].shape, (self.batch_size, self.hid_dim,))
        self.assertEqual(hid[3].shape, (self.batch_size, self.hid_dim,))

    def test_forward_no_conv(self):
        output, _ = self.model(self.input, self.hid_0)
        self.assertEqual(output.shape, (self.batch_size, self.inp_dim))
        
    def test_forward_with_conv(self):
        output, _ = self.model(self.input, self.hid_0, use_conv=True)
        self.assertEqual(output.shape, (self.batch_size, self.inp_dim))

    def test_backward(self):
        criterion = torch.nn.MSELoss()
        
        target = torch.randn(self.batch_size, self.inp_dim)
        output, _ = self.model(self.input, self.hid_0)
        
        loss = criterion(output, target)
        loss.backward()

        # Check if gradients are computed for all parameters
        # with the possible exception of the causal conv
        for name, param in self.model.named_parameters():
            if 'causal_conv' in name: continue
            self.assertIsNotNone(param.grad)

if __name__ == '__main__':
    unittest.main()