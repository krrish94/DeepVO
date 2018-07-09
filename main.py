from __future__ import print_function, division
import torch
import sys
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import random as rn
import torch.nn as nn
import torch.optim as optim


# Files with definitions
import model
import args
import dataloader

# Get the command arguements and intitialze the dataloader
cmd = args.arguments;
dataloader = dataloader.Dataloader()
# Set the default tensor type
torch.set_default_tensor_type(torch.cuda.FloatTensor)
# dummy_inp = torch.cuda.FloatTensor(1,6,384,1280)
########################################################################
			### Train and Validation Function ###
########################################################################
def train():
	trainSeqs = dataloader.train_seqs_KITTI
	trajLength = range(dataloader.minFrame_KITTI,dataloader.maxFrame_KITTI)
	rn.shuffle(trainSeqs)
	rn.shuffle(trajLength)
	for seq in trainSeqs:
		for tl in trajLength:
			# get a random subsequence from 'seq' of length 'fl' : starting index, ending index
			stFrm, enFrm = dataloader.getSubsequence(seq,tl,cmd.dataset)
			# itterate over this subsequence and get the frame data.
			for frm1 in range(stFrm,enFrm):
				inp,axis,t = dataloader.getPairFrameInfo(frm1,frm1+1,seq,cmd.dataset)
				# Forward, compute loss and backprop
				if cmd.loadModel!= "none":
					output_cnn = cnn.forward(inp)
					output_r, output_t = lstm.forward(output_cnn)
					loss_r = criterion(output_r,axis)
					loss_t = criterion(output_t,t)
					
				else:
					output_r,output_t = m.forward(inp)






def valid():
	trainSeqs = dataloader.train_seqs_KITTI
	trajLength = range(dataloader.minFrame_KITTI,dataloader.maxFrame_KITTI)
	rn.shuffle(trainSeqs)
	rn.shuffle(trajLength)

	for seq in trainSeqs:
		for tl in trajLength:
			# get a random subsequence from 'seq' of length 'fl' : starting index, ending index
			stFrm, enFrm = dataloader.getSubsequence(seq,tl,cmd.dataset)
			# itterate over this subsequence and get the frame data.
			for frm1 in range(stFrm,enFrm):
				inp,axis,t = dataloader.getPairFrameInfo(frm1,frm1+1,seq,cmd.dataset)
				# Forward, compute loss and backprop
				if cmd.loadModel!= "none":
					output_cnn = cnn.forward(inp)
					output_r, output_t = lstm.forward(output_cnn)
					loss_r = criterion(output_r,axis)
					loss_t = criterion(output_t,t)
				else:
					output_r,output_t = m.forward(inp)




########################################################################
			### Model Definition + Weight loading ###
########################################################################


if cmd.loadModel == 'none':
	# No pretrained weights, get the entire DeepVO model.
	print(" No weights specified")
	m = model.Net_DeepVO()
	m.cuda()
	# op = m.forward(dummy_inp)
	# print(op)
	# print(op.shape)

else:
	print('==> Loading pretrained weights')
	# Path for model(ordered dict) or checkpoint
	path = cmd.loadModel
	if cmd.modelType == 'caffe':
		print(" From the caffe model")
		flownetModel = torch.load(path)
		cnn = model.Net_CNN()
		cnn = model.copyWeights(cnn,flownetModel,cmd.modelType)

	elif cmd.modelType == 'checkpoint_wob':

		print(' From without batchnorm checkpoint')
		checkpoint = torch.load(path)
		flownetModel = checkpoint['state_dict']
		cnn = model.Net_CNN()
		cnn = model.copyWeights(cnn,flownetModel,cmd.modelType)

	else:
		print(' From with batchnorm checkpoint')
		checkpoint = torch.load(path)
		flownetModel = checkpoint['state_dict']
		cnn = model.Net_CNN_BN()
    	cnn = model.copyWeights(cnn,flownetModel,cmd.modelType)
    
	cnn.cuda()
	lstm = model.Net_LSTM()
	lstm.cuda()

	# op1 = cnn.forward(dummy_inp)
	# op2 = lstm.forward(op1)
	# print(op2)
	# print(op2.shape)


########################################################################
				### Criterion and optimizer ###
########################################################################

criterion = nn.MSELoss()
modelParameters = list(cnn.parameters()) + list(lstm.parameters())
if cmd.optMethod == "adam":
	optimizer = optim.Adam(modelParameters, lr = cmd.lr,  weight_decay = cmd.weightDecay,amsgrad = False)
elif cmd.optMethod == "sgd":
	optimizer = optim.SGD(modelParameters, lr = cmd.lr, momentum = cmd.momentum, weight_decay = cmd.weightDecay, nesterov = False)
else:
	optimizer = optim.Adagrad(modelParameters, lr = cmd.lr, lr_decay = cmd.lrDecay , weight_decay = cmd.weightDecay)


########################################################################
				###  Main loop ###
########################################################################




for epoch in range(cmd.nepochs):
	print("==> Starting epoch: "  + str(epoch+1) + "/" + str(cmd.nepochs))

	#train 
	train()

	# validation
	valid()

	


	

