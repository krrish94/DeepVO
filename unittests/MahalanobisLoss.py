"""
An implementation of the Mahalanobis norm as a loss criterion
"""

import torch
import torch.nn as nn
from torch.autograd import Function, Variable


class MahalanobisLoss(Function):

	# Both forward and backward here are implemented as @staticmethods, following the
	# example at https://pytorch.org/docs/master/notes/extending.html

	@staticmethod
	def forward(ctx, pred, gt, infoMat = None):

		ctx.save_for_backward(pred, gt, infoMat)
		diff = torch.add(pred, -gt)
		if infoMat is not None:
			mahal = 0.5 * torch.mm(diff.t(), torch.mm(infoMat, diff))
		else:
			mahal = 0.5 * torch.mm(diff.t(), diff)
		return mahal

	# We only compute gradient wrt outputs of the forward pass (i.e., wrt 'mahal'), since
	# there are no learnable parameters

	@staticmethod
	def backward(ctx, grad_output):

		# Unpack saved tensors
		pred, gt, infoMat = ctx.saved_tensors
		grad_pred = grad_gt = grad_infoMat  = None

		# Return values
		if ctx.needs_input_grad[0]:
			grad_pred = None
		if ctx.needs_input_grad[1]:
			grad_gt = None
		if ctx.needs_input_grad[2]:
			grad_infoMat = None

		return grad_pred, grad_gt, grad_infoMat
