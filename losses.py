"""
Custom loss functions go here
"""

import torch


# Mahalanobis distance implementation
# Inputs: prediction, gt (usually 1 x 6 vectors)
# Optional input: infoMat (usually a 6 x 6 information (i.e., inverse covariance) matrix)
# Output: a scalar quantity denoting the squared Mahalanobis distance
def MahalanobisLoss(pred, gt, infoMat = None):
	z = pred - gt
	if infoMat is not None:
		pass
		# print(torch.mm(z, torch.mm(infoMat, z.t())))
	else:
		pass
		# print(torch.mm(z, z.t()))
	# print(torch.mm(z, torch.mm(infoMat, z.t())))
	inv_sigma_z = torch.mm(infoMat, z.t())
	return torch.mm(z, inv_sigma_z)[0]
	# return torch.mm(z, z.t())
