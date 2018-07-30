# Unit test for helpers.py

import sys
import os
sys.path.append(str(os.path.join(os.path.dirname(sys.argv[0]), '..')))

from Model import DeepVO
import time
import torch
from torch.utils.data import DataLoader


if __name__ == '__main__':

	torch.set_default_tensor_type('torch.cuda.FloatTensor')
	# deepVO = DeepVO(numLSTMCells = 2, hidden_units_LSTM = [1024, 1024])
	deepVO = DeepVO(numLSTMCells = 3, hidden_units_LSTM = [1024, 1024, 512])
	deepVO = deepVO.cuda()
	inp = torch.rand(1, 6, 1280, 384).float().cuda()
	r, t = deepVO.forward(inp)
	deepVO.detach_LSTM_hidden()
