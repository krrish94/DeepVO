# Unit test for MahalanobisLoss.py

import sys
import os
sys.path.append(str(os.path.join(os.path.dirname(sys.argv[0]), '..')))

# from MahalanobisLoss import MahalanobisLoss
# from Model import DeepVO
import time
import torch
from torch.autograd import Variable, Function
import torch.nn
# from torch.utils.data import DataLoader

def mahal(pred, gt, infoMat = None):
	z = pred - gt
	if infoMat is not None:
		print(torch.mm(z, torch.mm(infoMat, z.t())))
	else:
		print(torch.mm(z, z.t()))
	return torch.mm(z, z.t())


if __name__ == '__main__':

	torch.set_default_tensor_type('torch.cuda.FloatTensor')
	# deepVO = DeepVO(numLSTMCells = 3, hidden_units_LSTM = [1024, 1024, 512])
	# deepVO = deepVO.cuda()
	# inp = torch.rand(1, 6, 1280, 384).float().cuda()
	# r, t = deepVO.forward(inp)
	# deepVO.detach_LSTM_hidden()
	
	# loss = MahalanobisLoss()
	infoMat = torch.FloatTensor([[100, 10, 10], [10, 100, 10], [10, 10, 100]])
	pred = Variable(torch.rand(1,6).float().cuda(), requires_grad = True)
	for i in range(50):
		gt = torch.ones(1,6).float().cuda()
		z = mahal(pred, gt, torch.eye(6))
		z.backward()
		# pred = pred -0.1 * pred.grad.data
		pred_up = pred - 0.1 * pred.grad.data
		pred.grad.zero_()
		pred.data = pred_up.data
	print(pred)
