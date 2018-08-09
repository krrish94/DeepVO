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
	inv_sigma_z = torch.mm(infoMat, z.t())
	return torch.mm(z, inv_sigma_z)[0]
