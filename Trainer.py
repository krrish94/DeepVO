"""
Trainer class. Handles training and validation
"""

from helpers import get_gpu_memory_map
from KITTIDataset import KITTIDataset
from Model import DeepVO
import numpy as np
import os
import sys
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm, trange


class Trainer():

	def __init__(self, args, epoch, model, train_set, val_set, loss_fn, optimizer, scheduler = None, \
		gradClip = None):


		# Commandline arguments
		self.args = args

		# Maximum number of epochs to train for
		self.maxEpochs = self.args.nepochs
		# Current epoch (initally set to -1)
		self.curEpoch = epoch

		# Model to train
		self.model = model

		# Train and validataion sets (Dataset objects)
		self.train_set = train_set
		self.val_set = val_set

		# Loss function
		if self.args.outputParameterization == 'mahalanobis':
			self.loss_fn = loss_fn
		else:
			self.loss_fn = nn.MSELoss(reduction = 'sum')

		# Variables to hold loss
		if self.args.outputParameterization == 'mahalanobis':
			self.loss = torch.zeros(1, dtype = torch.float32).cuda()
		else:
			self.loss_rot = Variable(torch.zeros(1, dtype = torch.float32).cuda(), requires_grad = False)
			self.loss_trans = Variable(torch.zeros(1, dtype = torch.float32).cuda(), requires_grad = False)
			self.loss = torch.zeros(1, dtype = torch.float32).cuda()

		# Optimizer
		self.optimizer = optimizer

		# Scheduler
		self.scheduler = scheduler

		# Flush gradient buffers before beginning training
		self.model.zero_grad()

		# Keep track of number of iters (useful for tensorboardX visualization)
		self.iters = 0


	# Train for one epoch
	def train(self):

		# Switch model to train mode
		self.model.train()

		# Check if maxEpochs have elapsed
		if self.curEpoch >= self.maxEpochs:
			print('Max epochs elapsed! Returning ...')
			return

		# Increment iters
		self.iters += 1

		# Variables to store stats
		rotLosses = []
		transLosses = []
		totalLosses = []
		rotLoss_seq = []
		transLoss_seq = []
		totalLoss_seq = []

		# Handle debug mode here
		if self.args.debug is True:
			numTrainIters = self.args.debugIters
		else:
			numTrainIters = len(self.train_set)

		# Initialize a variable to hold the number of sampes in the current batch
		# Here, 'batch' refers to the length of a subsequence that can be processed
		# before performing a 'detach' operation
		elapsedBatches = 0

		# Choose a generator (for iterating over the dataset, based on whether or not the 
		# sbatch flag is set to True). If sbatch is True, we're probably running on a cluster
		# and do not want an interactive output. So, could suppress tqdm and print statements
		if self.args.sbatch is True:
			gen = range(numTrainIters)
		else:
			gen = trange(numTrainIters)

		# Run a pass of the dataset
		for i in gen:

			if self.args.profileGPUUsage is True:
				gpu_memory_map = get_gpu_memory_map()
				tqdm.write('GPU usage: ' + str(gpu_memory_map[0]), file = sys.stdout)

			# Get the next frame
			inp, rot_gt, trans_gt, _, _, _, endOfSeq = self.train_set[i]

			# Feed it through the model
			rot_pred, trans_pred = self.model.forward(inp)

			# Compute loss
			# self.loss_rot += self.args.scf * self.loss_fn(rot_pred, rot_gt)
			if self.args.outputParameterization == 'mahalanobis':
				# Compute a mahalanobis norm on the output 6-vector
				# Note that, although we seem to be computing loss only over rotation variables
				# rot_pred and rot_gt are now 6-vectors that also include translation variables.
				self.loss += self.loss_fn(rot_pred, rot_gt, self.train_set.infoMat)
				tmpLossVar = Variable(torch.mm(rot_pred - rot_gt, torch.mm(self.train_set.infoMat, (rot_pred - rot_gt).t())), requires_grad = False).detach().cpu().numpy()
				# tmpLossVar = Variable(torch.dist(rot_pred, rot_gt) ** 2, requires_grad = False).detach().cpu().numpy()
				totalLosses.append(tmpLossVar[0])
				totalLoss_seq.append(tmpLossVar[0])
			else:
				curloss_rot = Variable(self.args.scf * (torch.dist(rot_pred, rot_gt) ** 2), requires_grad = False)
				curloss_trans = Variable(torch.dist(trans_pred, trans_gt) ** 2, requires_grad = False)
				self.loss_rot += curloss_rot
				self.loss_trans += curloss_trans

				if np.random.normal() < -0.9:
					tqdm.write('rot: ' + str(rot_pred.data) + ' ' + str(rot_gt.data), file = sys.stdout)
					tqdm.write('trans: ' + str(trans_pred.data) + ' ' + str(trans_gt.data), file = sys.stdout)

				self.loss += sum([self.args.scf * self.loss_fn(rot_pred, rot_gt), \
					self.loss_fn(trans_pred, trans_gt)])
				# self.loss = self.loss_fn(rot_pred, rot_gt)

				# # Compute gradients	# ???
				# self.loss = sum([self.args.scf * self.loss_fn(rot_pred, rot_gt), \
				# 	self.loss_fn(trans_pred, trans_gt)])
				# self.loss.backward()
				# # self.model.zero_grad()
				# self.model.detach_LSTM_hidden()

				# Store losses (for further analysis)
				# curloss_rot = (self.args.scf * self.loss_fn(rot_pred, rot_gt)).detach().cpu().numpy()
				# curloss_trans = (self.loss_fn(trans_pred, trans_gt)).detach().cpu().numpy()
				curloss_rot = curloss_rot.detach().cpu().numpy()
				curloss_trans = curloss_trans.detach().cpu().numpy()
				rotLosses.append(curloss_rot)
				transLosses.append(curloss_trans)
				totalLosses.append(curloss_rot + curloss_trans)
				rotLoss_seq.append(curloss_rot)
				transLoss_seq.append(curloss_trans)
				totalLoss_seq.append(curloss_rot + curloss_trans)

			# Handle debug mode here. Force execute the below if statement in the
			# last debug iteration
			if self.args.debug is True:
				if i == numTrainIters - 1:
					endOfSeq = True

			elapsedBatches += 1
			
			# if endOfSeq is True:
			if elapsedBatches >= self.args.trainBatch or endOfSeq is True:

				elapsedBatches = 0

				# # L2-Regularization				
				# if self.args.gamma > 0.0:
				# 	# Regularization for network weights
				# 	l2_reg = None
				# 	for W in self.model.parameters():
				# 		if l2_reg is None:
				# 			l2_reg = W.norm(2)
				# 		else:
				# 			l2_reg = l2_reg + W.norm(2)
				# 	self.loss = sum([self.weightRegularizer * l2_reg, self.loss])

				# # L1-Regularization
				# if self.args.gamma > 0.0:
				# 	l1_crit = nn.L1Loss(size_average = False)
				# 	reg_loss = None
				# 	for param in self.model.parameters():
				# 		reg_loss += l1_crit(param)
				# 	self.loss = sum([self.gamma * reg_loss, self.loss])

				# Regularize only LSTM(s)
				if self.args.gamma > 0.0:
					paramsDict = self.model.state_dict()
					# print(paramsDict.keys())
					if self.args.numLSTMCells == 1:
						reg_loss = None
						reg_loss = paramsDict['lstm1.weight_ih'].norm(2)
						reg_loss += paramsDict['lstm1.weight_hh'].norm(2)
						reg_loss += paramsDict['lstm1.bias_ih'].norm(2)
						reg_loss += paramsDict['lstm1.bias_hh'].norm(2)
					else:
						reg_loss = None
						reg_loss = paramsDict['lstm2.weight_ih'].norm(2)
						reg_loss += paramsDict['lstm2.weight_hh'].norm(2)
						reg_loss += paramsDict['lstm2.bias_ih'].norm(2)
						reg_loss += paramsDict['lstm2.bias_hh'].norm(2)
						reg_loss += paramsDict['lstm2.weight_ih'].norm(2)
						reg_loss += paramsDict['lstm2.weight_hh'].norm(2)
						reg_loss += paramsDict['lstm2.bias_ih'].norm(2)
						reg_loss += paramsDict['lstm2.bias_hh'].norm(2)
					self.loss = sum([self.args.gamma * reg_loss, self.loss])

				# Print stats
				if self.args.outputParameterization != 'mahalanobis':
					tqdm.write('Rot Loss: ' + str(np.mean(rotLoss_seq)) + ' Trans Loss: ' + \
						str(np.mean(transLoss_seq)), file = sys.stdout)
				else:
					tqdm.write('Total Loss: ' + str(np.mean(totalLoss_seq)), file = sys.stdout)
				rotLoss_seq = []
				transLoss_seq = []
				totalLoss_seq = []

				# Compute gradients	# ???
				self.loss.backward()

				# Monitor gradients
				l = 0
				# for p in self.model.parameters():
				# 	if l in [j for j in range(18,26)] + [j for j in range(30,34)]:
				# 		print(p.shape, 'GradNorm: ', p.grad.norm())
				# 	l += 1
				paramList = list(filter(lambda p : p.grad is not None, [param for param in self.model.parameters()]))
				totalNorm = sum([(p.grad.data.norm(2.) ** 2.) for p in paramList]) ** (1. / 2)
				tqdm.write('gradNorm: ' + str(totalNorm.item()))

				# Perform gradient clipping, if enabled
				if self.args.gradClip is not None:
					torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.gradClip)

				# Update parameters
				self.optimizer.step()

				# If it's the end of sequence, reset hidden states
				if endOfSeq is True:
					self.model.reset_LSTM_hidden()
				self.model.detach_LSTM_hidden()	# ???

				# Reset loss variables
				self.loss_rot = torch.zeros(1, dtype = torch.float32).cuda()
				self.loss_trans = torch.zeros(1, dtype = torch.float32).cuda()
				self.loss = torch.zeros(1, dtype = torch.float32).cuda()

				# Flush gradient buffers for next forward pass
				self.model.zero_grad()

		# Return loss logs for further analysis
		if self.args.outputParameterization == 'mahalanobis':
			return [], [], totalLosses
		else:
			return rotLosses, transLosses, totalLosses


	# Run one epoch of validation
	def validate(self):

		# Switch model to eval mode
		self.model.eval()

		# Run a pass of the dataset
		traj_pred = None

		# Variables to store stats
		rotLosses = []
		transLosses = []
		totalLosses = []
		rotLoss_seq = []
		transLoss_seq = []
		totalLoss_seq = []

		# Handle debug switch here
		if self.args.debug is True:
			numValIters = self.args.debugIters
		else:
			numValIters = len(self.val_set)

		# Choose a generator (for iterating over the dataset, based on whether or not the 
		# sbatch flag is set to True). If sbatch is True, we're probably running on a cluster
		# and do not want an interactive output. So, could suppress tqdm and print statements
		if self.args.sbatch is True:
			gen = range(numValIters)
		else:
			gen = trange(numValIters)

		for i in gen:

			if self.args.profileGPUUsage is True:
				gpu_memory_map = get_gpu_memory_map()
				tqdm.write('GPU usage: ' + str(gpu_memory_map[0]), file = sys.stdout)

			# Get the next frame
			inp, rot_gt, trans_gt, seq, frame1, frame2, endOfSeq = self.val_set[i]
			metadata = np.concatenate((np.asarray([seq]), np.asarray([frame1]), np.asarray([frame2])))
			metadata = np.reshape(metadata, (1, 3))

			# Feed it through the model
			rot_pred, trans_pred = self.model.forward(inp)

			if self.args.outputParameterization == 'mahalanobis':
				if traj_pred is None:
					traj_pred = np.concatenate((metadata, rot_pred.data.cpu().numpy()), axis = 1)
				else:
					cur_pred = np.concatenate((metadata, rot_pred.data.cpu().numpy()), axis = 1)
					traj_pred = np.concatenate((traj_pred, cur_pred), axis = 0)
			else:
				if traj_pred is None:
					traj_pred = np.concatenate((metadata, rot_pred.data.cpu().numpy(), \
						trans_pred.data.cpu().numpy()), axis = 1)
				else:
					cur_pred = np.concatenate((metadata, rot_pred.data.cpu().numpy(), \
						trans_pred.data.cpu().numpy()), axis = 1)
					traj_pred = np.concatenate((traj_pred, cur_pred), axis = 0)

			# Store losses (for further analysis)
			if self.args.outputParameterization == 'mahalanobis':
				# rot_pred and rot_gt are 6-vectors here, and they include translations too
				tmpLossVar = self.loss_fn(rot_pred, rot_gt, self.train_set.infoMat).detach().cpu().numpy()
				totalLosses.append(tmpLossVar[0])
				totalLoss_seq.append(tmpLossVar[0])
			else:
				curloss_rot = (self.args.scf * self.loss_fn(rot_pred, rot_gt)).detach().cpu().numpy()
				curloss_trans = (self.loss_fn(trans_pred, trans_gt)).detach().cpu().numpy()
				rotLosses.append(curloss_rot)
				transLosses.append(curloss_trans)
				totalLosses.append(curloss_rot + curloss_trans)
				rotLoss_seq.append(curloss_rot)
				transLoss_seq.append(curloss_trans)
				totalLoss_seq.append(curloss_rot + curloss_trans)

			# Detach hidden states and outputs of LSTM
			self.model.detach_LSTM_hidden()

			if endOfSeq is True:

				# Print stats
				if self.args.outputParameterization != 'mahalanobis':
					tqdm.write('Rot Loss: ' + str(np.mean(rotLoss_seq)) + ' Trans Loss: ' + \
						str(np.mean(transLoss_seq)), file = sys.stdout)
				else:
					tqdm.write('Total Loss: ' + str(np.mean(totalLoss_seq)), file = sys.stdout)
				rotLoss_seq = []
				transLoss_seq = []
				totalLoss_seq = []

				# Write predicted trajectory to file
				saveFile = os.path.join(self.args.expDir, 'plots', 'traj', str(seq).zfill(2), \
					'traj_' + str(self.curEpoch).zfill(3) + '.txt')
				np.savetxt(saveFile, traj_pred, newline = '\n')
				
				# Reset variable, to store new trajectory later on
				traj_pred = None
				
				# Detach LSTM hidden states
				self.model.detach_LSTM_hidden()

				# Reset LSTM hidden states
				self.model.reset_LSTM_hidden()


		# Return loss logs for further analysis
		if self.args.outputParameterization == 'mahalanobis':
			return [], [], totalLosses
		else:
			return rotLosses, transLosses, totalLosses
