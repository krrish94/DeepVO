from __future__ import print_function, division
from torch.autograd import Variable as V
from drawPlot import plotSequences

import sys
import os
import torch
import numpy as np
import random as rn
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



# Files with definitions
import model
import args
import dataloader
import time


# Get the command arguements and intitialze the dataloader
cmd = args.arguments;
dataloader = dataloader.Dataloader()
# Set the default tensor type
torch.set_default_tensor_type(torch.cuda.FloatTensor)

if not os.path.exists("/u/sharmasa/Documents/DeepVO/exp/" + cmd.dataset):
	os.makedirs("/u/sharmasa/Documents/DeepVO/exp/" + cmd.dataset)

# Make the dirctory structure to save models and plots
if not os.path.exists("/u/sharmasa/Documents/DeepVO/exp/" + cmd.dataset + "/" + cmd.expID):
    os.makedirs("/u/sharmasa/Documents/DeepVO/exp/" + cmd.dataset + "/" + cmd.expID)
    os.makedirs("/u/sharmasa/Documents/DeepVO/exp/" + cmd.dataset +"/" + cmd.expID + "/plots")
    os.makedirs("/u/sharmasa/Documents/DeepVO/exp/" + cmd.dataset +"/" + cmd.expID + "/models")

    os.makedirs("/u/sharmasa/Documents/DeepVO/exp/" + cmd.dataset +"/" + cmd.expID + "/plots/traj")
    os.makedirs("/u/sharmasa/Documents/DeepVO/exp/" + cmd.dataset +"/" + cmd.expID + "/plots/loss")
    
    for seq in dataloader.total_seqs_KITTI:
    	os.makedirs("/u/sharmasa/Documents/DeepVO/exp/" + cmd.dataset + "/" + cmd.expID + "/plots/traj/" + str(seq).zfill(2))


# Save all the command line arguements in a text file in the experiment directory.
cmdFile = open("/u/sharmasa/Documents/DeepVO/exp/" + cmd.dataset + "/" + cmd.expID + "/args.txt", 'w')
for arg in vars(cmd):
	cmdFile.write(arg + " " + str(getattr(cmd,arg)) + "\n")
cmdFile.close()
########################################################################
			   			### Train and validation Function ###
########################################################################
def train(epoch):

	#Switching to train mode
	deepVO.train()
	
	trainSeqs = dataloader.train_seqs_KITTI
	trajLength = range(dataloader.minFrame_KITTI,dataloader.maxFrame_KITTI, rn.randint(5,10))

	rn.shuffle(trainSeqs)
	rn.shuffle(trajLength)

	itt_T_Loss=0.0
	itt_R_Loss=0.0

	num_itt=0;
	
	avgRotLoss=[];
	avgTrLoss=[];

	

	for seq in trainSeqs:
		for tl in trajLength:
			# get a random subsequence from 'seq' of length 'fl' : starting index, ending index
			stFrm, enFrm = dataloader.getSubsequence(seq,tl,cmd.dataset)
			# itterate over this subsequence and get the frame data.
			flag=0;
			print("Sequence : ", seq, "start frame : ", stFrm, "end frame : ", enFrm)
			for frm1 in range(stFrm,enFrm):
		
				inp,axis,t = dataloader.getPairFrameInfo(frm1,frm1+1,seq,cmd.dataset)

				
				# Forward, compute loss and backprop
				deepVO.zero_grad()
				output_r, output_t = deepVO.forward(inp,flag)
				
				loss_r = criterion(output_r,axis)
				loss_t = criterion(output_t,t)

				# Net loss : rotation + translation
				loss = loss_r + cmd.scf*loss_t;
				loss = V(loss,requires_grad=True)
				loss.backward()


				# There are two ways to save the hidden states for temporal use :
				# WAY 1. use retain graph = true while doing backward pass , for all the passes except the last one where the (sub) sequence ends. 
				#    This will retain the graph for as long as you want and will terminate when you call loss.backward(retain_graph=False).
				#    Cons : Very slow, high memory usage
				# WAY 2. detach all the hidden states/cell states that are useful for temporal information before the graph is discarded after each forward pass.
				      # Done in forward pass of the model. fast and not too much of memory consumption.

				# loss = V(loss,requires_grad=True)
				# if frm1 != enFrm-1:
				# 	loss.backward(retain_graph=True)
				# else:
				# 	loss.backward(retain_graph=False)
				optimizer.step()



				# Save rotation and translation loss values
				itt_R_Loss = (itt_R_Loss*num_itt + loss_r.item())/(num_itt+1)
				itt_T_Loss = (itt_T_Loss*num_itt + loss_t.item())/(num_itt+1)

				flag=1;
				num_itt =num_itt+1

				if num_itt == cmd.iterations:
					avgRotLoss.append(itt_R_Loss)
					avgTrLoss.append(itt_T_Loss)
					num_itt=0;
					itt_T_Loss=0.0
					itt_R_Loss=0.0

			
					
	# Save plot for loss					
	fig_r,ax_r = plt.subplots(1)
	ax_r.plot(avgRotLoss,'r')
	plt.ylabel("Rotation Loss")
	plt.xlabel("Per " + str(cmd.iterations) + " iterations in one epoch")
	fig_r.savefig("/u/sharmasa/Documents/DeepVO/exp/" + cmd.dataset + "/" + cmd.expID + "/plots/loss/" + "rotLoss_epoch_" + str(epoch))
	
	fig_t,ax_t = plt.subplots(1)
	ax_t.plot(avgTrLoss,'g')
	plt.ylabel("Translation Loss")
	plt.xlabel("Per " + str(cmd.iterations) + " iterations in one epoch")
	fig_t.savefig("/u/sharmasa/Documents/DeepVO/exp/" + cmd.dataset + "/" + cmd.expID + "/plots/loss/" + "transLoss_epoch_" + str(epoch))

	return sum(avgRotLoss)/float(len(avgRotLoss)), sum(avgTrLoss)/float(len(avgTrLoss))


def validate(epoch,tag="valid"):

	# Switching to evaluation mode
	deepVO.eval()

	# In validation will predicit the loss between every two successive frames in all the training / validation sequences.
	if tag=="train":
		validSeqs = dataloader.train_seqs_KITTI
	else:
		validSeqs = dataloader.test_seqs_KITTI

	# For the entire validation set
	avgRotLoss = []
	avgTrLoss = []
	# avgRotLoss = 0.0
	# avgTrLoss = 0.0

	
	for idx, seq in enumerate(validSeqs):
		seqLength = len(os.listdir("/data/milatmp1/sharmasa/"+ cmd.dataset + "/dataset/sequences/" + str(seq).zfill(2) + "/image_2/"))
		# To store the entire estimated trajector 
		seq_traj = np.zeros([seqLength-1,6])

		# Loss for each sequence
		# avgR_Loss_seq = 0.0;
		# avgT_Loss_seq = 0.0;
		avgR_Loss_seq = []
		avgT_Loss_seq = []
		
		
		flag = 0;

		for frame1 in range(seqLength-1):
			#print(frame1)
			inp,axis,t = dataloader.getPairFrameInfo(frame1,frame1+1,seq,cmd.dataset)
			
			output_r, output_t = deepVO.forward(inp,flag)
			# Outputs come in form of torch tensor. Convert to numpy.
			seq_traj[frame1] = np.append(output_r.data.cpu().numpy(),output_t.data.cpu().numpy(),axis=1)

			loss_r = criterion(output_r,axis)
			loss_t = criterion(output_t,t)
			# avgR_Loss_seq = (avgR_Loss_seq*frame1 + loss_r.item())/(frame1+1)
			# avgT_Loss_seq = (avgT_Loss_seq*frame1 + loss_t.item())/(frame1+1)
			avgR_Loss_seq.append(loss_r.item())
			avgT_Loss_seq.append(loss_t.item())

			
			flag = 1


		# Plot the trajectory of that sequence
		# if tag == "valid":
		# 	plotSequences(seq,seqLength,seq_traj,cmd.dataset,cmd)


		# Save the trajectory to text file of that sequence
		filepath = "/u/sharmasa/Documents/DeepVO/exp/" + cmd.dataset + "/" + cmd.expID + "/plots/traj/" +  str(seq).zfill(2) + "/traj_" + str(epoch) +".txt"
		np.savetxt(filepath, seq_traj, newline="\n")

		# Save rotation and translation loss of that sequence
		fig_r,ax_r = plt.subplots(1)
		ax_r.plot(avgR_Loss_seq,'r')
		plt.ylabel("Rotation Loss")
		plt.xlabel("Per pair of frames")
		fig_r.savefig("/u/sharmasa/Documents/DeepVO/exp/" + cmd.dataset + "/" + cmd.expID + "/plots/loss/" + "seq_ " + str(seq) + "_rotLoss_valid_epoch_" + str(epoch))

		fig_t,ax_t = plt.subplots(1)
		ax_t.plot(avgT_Loss_seq,'g')
		plt.ylabel("Translation Loss")
		plt.xlabel("Per pair of frames")
		fig_t.savefig("/u/sharmasa/Documents/DeepVO/exp/" + cmd.dataset + "/" + cmd.expID + "/plots/loss/" + "seq_ " + str(seq) + "_trans_Loss_valid_epoch_" + str(epoch))

		# avgRotLoss = (avgRotLoss*idx + avgR_Loss_seq)/(idx+1)
		# avgTrLoss = (avgTrLoss*idx + avgT_Loss_seq)/(idx+1)

		avgRotLoss.append(np.mean(avgR_Loss_seq))
		avgTrLoss.append(np.mean(avgT_Loss_seq))






	# Return average rotation and translation loss for all the validation sequences.
	return avgRotLoss,avgTrLoss				

########################################################################
			  ### Model Definition + Weight loading ###
########################################################################

# Get the definition of the model
if cmd.modelType !="checkpoint_wb":
# Model definition without batchnorm
	deepVO = model.Net_DeepVO_WOB()

else:

	# Model definition with batchnorm
	deepVO = model.Net_DeepVO_WB()

# Copy weights
if cmd.loadModel != "none":
	
	print('==> Loading pretrained weights')
	# From caffe model
	path = cmd.loadModel
	if cmd.modelType =="caffe":
		print(" From the caffe model")
		flownetModel = torch.load(path)
	
	# From checkpoint : without batchnorm
	elif cmd.modelType == "checkpoint_wob":
		print(' From without batchnorm checkpoint')
		checkpoint = torch.load(path)
		flownetModel = checkpoint['state_dict']
		# From checkpoint : with batchnorm
	else:
		print(' From with batchnorm checkpoint')
		checkpoint = torch.load(path)
		flownetModel = checkpoint['state_dict']

	deepVO = model.copyWeights(deepVO,flownetModel,cmd.modelType)

	# For the linear layers of the model
	if cmd.initType == "xavier":
		deepVO.init_weights()

	deepVO.cuda()
	print(' Loaded weights !!')


########################################################################
				   ### Criterion and optimizer ###
########################################################################

criterion = nn.MSELoss()

if cmd.optMethod == "adam":
	optimizer = optim.Adam(deepVO.parameters(), lr = cmd.lr,  weight_decay = cmd.weightDecay,amsgrad = False)
elif cmd.optMethod == "sgd":
	optimizer = optim.SGD(deepVO.parameters(), lr = cmd.lr, momentum = cmd.momentum, weight_decay = cmd.weightDecay, nesterov = False)
else:
	optimizer = optim.Adagrad(deepVO.parameters(), lr = cmd.lr, lr_decay = cmd.lrDecay , weight_decay = cmd.weightDecay)


########################################################################
				            ###  Main loop ###
########################################################################
r_tr=[];
t_tr=[]
r_val=[]
t_val=[]

for epoch in range(cmd.nepochs):

	print("================> Starting epoch: "  + str(epoch+1) + "/" + str(cmd.nepochs))
	
	# Average loss over one training epoch	
	r_trLoss , t_trLoss=train(epoch)
	r_tr.append(r_trLoss)
	t_tr.append(t_trLoss)

	# Average loss over entire validation set, a list of loss for all sequences
	r_valLoss, t_valLoss = validate(epoch,"valid")
	r_val.append(np.mean(r_valLoss))
	t_val.append(np.mean(t_valLoss))

	# After all the epochs plot the translation and rotation loss  w.r.t. epochs
	fig_r,ax_r = plt.subplots(1)
	ax_r.plot(r_valLoss)
	plt.ylabel("Rotation Loss : validation")
	plt.xlabel("Per sequence")
	fig_r.savefig("/u/sharmasa/Documents/DeepVO/exp/" + cmd.dataset + "/" + cmd.expID + "/" + "val_rotLoss" + str(epoch))
	
	fig_t,ax_t = plt.subplots(1)
	ax_t.plot(t_valLoss)
	plt.ylabel("Translation Loss : validation")
	plt.xlabel("Per sequence")
	fig_t.savefig("/u/sharmasa/Documents/DeepVO/exp/" + cmd.dataset + "/" + cmd.expID + "/" + "val_trainLoss" + str(epoch))

	



# After all the epochs plot the translation and rotation loss  w.r.t. epochs
fig_r,ax_r = plt.subplots(1)
ax_r.plot(r_tr,label="train")
ax_r.plot(r_val,label="valid")
ax_r.legend()
plt.ylabel("Rotation Loss")
plt.xlabel("Per epoch")
fig_r.savefig("/u/sharmasa/Documents/DeepVO/exp/" + cmd.dataset + "/" + cmd.expID + "/" + "rotLoss")

fig_t,ax_t = plt.subplots(1)
ax_t.plot(t_tr,label =" train")
ax_t.plot(t_val,label="valid")
ax_t.legend()
plt.ylabel("Translation Loss")
plt.xlabel("Per epoch")
fig_t.savefig("/u/sharmasa/Documents/DeepVO/exp/" + cmd.dataset + "/" + cmd.expID + "/" + "trainLoss")
print("Done !!")

