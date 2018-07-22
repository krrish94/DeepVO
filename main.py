"""
Main script: Train and test DeepVO on the KITTI odometry benchmark
"""


from __future__ import print_function, division
import itertools
import random as rn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
import torch
from torch.autograd import Variable as V
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm, trange

# Other project files with definitions
import args
import dataloader
from drawPlot import plotSequences
import model

# Get the commandline arguements
cmd = args.arguments;

# Seed the RNGs (ensure deterministic outputs)
rn.seed(cmd.randomseed)
np.random.seed(cmd.randomseed)
torch.manual_seed(cmd.randomseed)
torch.cuda.manual_seed(cmd.randomseed)
torch.backends.cudnn.deterministic = True

# Intitialze the dataloader
dataloader = dataloader.Dataloader(cmd.datadir)
# Set the default tensor type
torch.set_default_tensor_type(torch.cuda.FloatTensor)

# if not os.path.exists("/u/sharmasa/Documents/DeepVO/exp/" + cmd.dataset):
# 	os.makedirs("/u/sharmasa/Documents/DeepVO/exp/" + cmd.dataset)

basedir = os.path.dirname(os.path.realpath(__file__))
if not os.path.exists(os.path.join(basedir, cmd.cachedir, cmd.dataset)):
	os.makedirs(os.path.join(basedir, cmd.cachedir, cmd.dataset))

# Make the dirctory structure to save models and plots
expDir = os.path.join(basedir, cmd.cachedir, cmd.dataset, cmd.expID)
if not os.path.exists(expDir):
	os.makedirs(expDir)
	print('Created dir: ', expDir)
if not os.path.exists(os.path.join(expDir, 'models')):
	os.makedirs(os.path.join(expDir, 'models'))
	print('Created dir: ', os.path.join(expDir, 'models'))
if not os.path.exists(os.path.join(expDir, 'plots', 'traj')):
	os.makedirs(os.path.join(expDir, 'plots', 'traj'))
	print('Created dir: ', os.path.join(expDir, 'plots', 'traj'))
if not os.path.exists(os.path.join(expDir, 'plots', 'loss')):
	os.makedirs(os.path.join(expDir, 'plots', 'loss'))
	print('Created dir: ', os.path.join(expDir, 'plots', 'loss'))
for seq in dataloader.total_seqs_KITTI:
	if not os.path.exists(os.path.join(expDir, 'plots', 'traj', str(seq).zfill(2))):
		os.makedirs(os.path.join(expDir, 'plots', 'traj', str(seq).zfill(2)))
		print('Created dir: ', os.path.join(expDir, 'plots', 'traj', str(seq).zfill(2)))
		
# if not os.path.exists("/u/sharmasa/Documents/DeepVO/exp/" + cmd.dataset + "/" + cmd.expID):
#     os.makedirs("/u/sharmasa/Documents/DeepVO/exp/" + cmd.dataset + "/" + cmd.expID)
#     os.makedirs("/u/sharmasa/Documents/DeepVO/exp/" + cmd.dataset +"/" + cmd.expID + "/plots")
#     os.makedirs("/u/sharmasa/Documents/DeepVO/exp/" + cmd.dataset +"/" + cmd.expID + "/models")

#     os.makedirs("/u/sharmasa/Documents/DeepVO/exp/" + cmd.dataset +"/" + cmd.expID + "/plots/traj")
#     os.makedirs("/u/sharmasa/Documents/DeepVO/exp/" + cmd.dataset +"/" + cmd.expID + "/plots/loss")
	
#     for seq in dataloader.total_seqs_KITTI:
#     	os.makedirs("/u/sharmasa/Documents/DeepVO/exp/" + cmd.dataset + "/" + cmd.expID + "/plots/traj/" + str(seq).zfill(2))


# Save all the command line arguements in a text file in the experiment directory.
cmdFile = open(os.path.join(expDir, 'args.txt'), 'w')
for arg in vars(cmd):
	cmdFile.write(arg + ' ' + str(getattr(cmd, arg)) + '\n')
cmdFile.close()
# cmdFile = open("/u/sharmasa/Documents/DeepVO/exp/" + cmd.dataset + "/" + cmd.expID + "/args.txt", 'w')
# for arg in vars(cmd):
# 	cmdFile.write(arg + " " + str(getattr(cmd,arg)) + "\n")
# cmdFile.close()

########################################################################
### Train and validation Functions ###
########################################################################
def train(epoch):

	#Switching to train mode
	deepVO.train()
	
	trainSeqs = dataloader.train_seqs_KITTI
	# trajLength = list(range(dataloader.minFrame_KITTI, dataloader.maxFrame_KITTI, \
	# 	rn.randint(5, 20)))
	trajLength = list(itertools.chain.from_iterable(itertools.repeat(x, 100) for x in [20]))
	# trajLength = [40]	# ???

	rn.shuffle(trainSeqs)
	rn.shuffle(trajLength)

	itt_T_Loss = 0.0
	itt_R_Loss = 0.0
	itt_tot_Loss = 0.0

	num_itt = 0
	
	avgRotLoss = []
	avgTrLoss = []
	avgTotalLoss = []
	

	for seq in trainSeqs:
		# for tl in trajLength:
		for tl in tqdm(trajLength, unit = 'seqs'):
			# get a random subsequence from 'seq' of length 'fl' : starting index, ending index
			stFrm, enFrm = dataloader.getSubsequence(seq, tl, cmd.dataset)
			# stFrm, enFrm = 0, 40	# ???
			# itterate over this subsequence and get the frame data.
			flag = 0
			# print("Sequence : ", seq, "start frame : ", stFrm, "end frame : ", enFrm)
			tqdm.write('Epoch: ' + str(epoch) + ' Sequence : ' + str(seq) + ' Start frame : ' \
				+ str(stFrm) + ' End frame : ' + str(enFrm), file = sys.stdout)
			for frm1 in range(stFrm, enFrm):

				inp, axis, t = dataloader.getPairFrameInfo(frm1, frm1+1, seq, cmd.dataset)
				# axis = torch.tensor([[1.0, 1.0, 1.0]])	# ???
				# t = torch.tensor([[1.0, 1.0, 1.0]])		# ???
				
				# Forward, compute loss and backprop
				deepVO.zero_grad()
				output_r, output_t = deepVO.forward(inp, flag)
				
				loss_r = criterion(output_r,axis)
				loss_t = cmd.scf * criterion(output_t,t)
				
				# Soln 1: This works ###
				# loss_r.backward(retain_graph = True)
				# if frm1 != enFrm-1:
				# 	loss_t.backward(retain_graph = True)
				# else:
				# 	loss_t.backward(retain_graph = False)
				
				# Soln 2:
				loss = sum([loss_r, loss_t])
				if frm1 != enFrm-1:
					loss.backward(retain_graph = True)
				else:
					loss.backward(retain_graph = False)

				# Perform gradient clipping
				if cmd.gradClip is not None:
					torch.nn.utils.clip_grad_norm_(deepVO.parameters(), cmd.gradClip)

				# Update model parameters
				optimizer.step()

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


				# Save rotation and translation loss values
				itt_R_Loss = (itt_R_Loss*num_itt + loss_r.item())/(num_itt+1)
				itt_T_Loss = (itt_T_Loss*num_itt + loss_t.item())/(num_itt+1)
				itt_tot_Loss = (itt_tot_Loss*num_itt + loss.item()) / (num_itt + 1)

				flag = 1
				num_itt = num_itt + 1

				if num_itt == cmd.iterations:
					avgRotLoss.append(itt_R_Loss)
					avgTrLoss.append(itt_T_Loss)
					avgTotalLoss.append(itt_tot_Loss)
					num_itt = 0
					itt_T_Loss = 0.0
					itt_R_Loss = 0.0
					itt_tot_Loss = 0.0

			# print('Rot Loss: ', str(itt_R_Loss), 'Trans Loss: ', str(itt_T_Loss))
			# print('Total Loss: ', str(itt_tot_Loss))
			tqdm.write('Rot Loss: ' + str(itt_R_Loss) + ' Trans Loss: ' + str(itt_T_Loss), 
				file = sys.stdout)
			tqdm.write('Total Loss: ' + str(itt_tot_Loss), file = sys.stdout)

					
	# Save plot for loss					
	fig_r,ax_r = plt.subplots(1)
	ax_r.plot(avgRotLoss,'r')
	plt.ylabel("Rotation Loss")
	plt.xlabel("Per " + str(cmd.iterations) + " iterations in one epoch")
	fig_r.savefig(os.path.join(expDir, 'plots', 'loss', 'rotLoss_epoch_' + str(epoch)))
	
	fig_t,ax_t = plt.subplots(1)
	ax_t.plot(avgTrLoss,'g')
	plt.ylabel("Translation Loss")
	plt.xlabel("Per " + str(cmd.iterations) + " iterations in one epoch")
	fig_t.savefig(os.path.join(expDir, 'plots', 'loss', 'transLoss_epoch_' + str(epoch)))

	fig_tot, ax_tot = plt.subplots(1)
	ax_tot.plot(avgTotalLoss, 'b')
	plt.ylabel('Total Loss')
	plt.xlabel('Per' + str(cmd.iterations) + ' iterations in one epoch')
	fig_tot.savefig(os.path.join(expDir, 'plots', 'loss', 'totalLoss_epoch_' + str(epoch)))

	if avgRotLoss == [] and avgTrLoss == [] and avgTotalLoss == []:
		return 0.0, 0.0, 0.0
	return sum(avgRotLoss)/float(len(avgRotLoss)), sum(avgTrLoss)/float(len(avgTrLoss)), sum(avgTotalLoss)/float(len(avgTotalLoss))


def validate(epoch, tag = 'valid'):

	# Switching to evaluation mode
	deepVO.eval()

	# In validation will predicit the loss between every two successive frames in all the training / validation sequences.
	if tag == 'train':
		validSeqs = dataloader.train_seqs_KITTI
	else:
		validSeqs = dataloader.test_seqs_KITTI

	# For the entire validation set
	avgRotLoss = []
	avgTrLoss = []
	avgTotalLoss = []


	for seq in tqdm(validSeqs, unit = 'sequences'):

		seqLength = len(os.listdir(os.path.join(cmd.datadir, 'sequences', str(seq).zfill(2), 'image_2')))
		# seqLength = len(os.listdir("/data/milatmp1/sharmasa/"+ cmd.dataset + "/dataset/sequences/" + str(seq).zfill(2) + "/image_2/"))
		# seqLength = 41	# ???
		# To store the entire estimated trajector 
		seq_traj = np.zeros([seqLength-1,6])

		# Loss for each sequence
		avgR_Loss_seq = []
		avgT_Loss_seq = []
		avgTotal_Loss_seq = []
				
		flag = 0;

		for frame1 in trange(seqLength-1):

			inp,axis,t = dataloader.getPairFrameInfo(frame1, frame1+1, seq,cmd.dataset)
			# axis = torch.tensor([[1.0, 1.0, 1.0]])	# ???
			# t = torch.tensor([[1.0, 1.0, 1.0]])		# ???
			
			output_r, output_t = deepVO.forward(inp, flag)

			# Outputs come in form of torch tensor. Convert to numpy.
			seq_traj[frame1] = np.append(output_r.data.cpu().numpy(), output_t.data.cpu().numpy(), axis = 1)

			loss_r = criterion(output_r,axis)
			loss_t = criterion(output_t,t)
			loss = sum([loss_r, cmd.scf * loss_t])
			
			avgR_Loss_seq.append(loss_r.item())
			avgT_Loss_seq.append(loss_t.item())
			avgTotal_Loss_seq.append(loss.item())
			
			flag = 1


		# print('Rot Loss: ', str(np.mean(avgR_Loss_seq)), 'Trans Loss: ', str(np.mean(avgT_Loss_seq)))
		# print('Total Loss: ', str(np.mean(avgTotal_Loss_seq)))
		tqdm.write('Rot Loss: ' + str(np.mean(avgR_Loss_seq)) + ' Trans Loss: ' + \
			str(np.mean(avgT_Loss_seq)), file = sys.stdout)
		tqdm.write('Total Loss: ' + str(np.mean(avgTotal_Loss_seq)), file = sys.stdout)

		# # Plot the trajectory of that sequence
		# if tag == "valid":
		# 	plotSequences(seq, seqLength, seq_traj, cmd.dataset, cmd)


		# Save the trajectory to text file of that sequence
		filepath = os.path.join(expDir, 'plots', 'traj', str(seq).zfill(2), 'traj_' + str(epoch) + '.txt')
		np.savetxt(filepath, seq_traj, newline="\n")

		# Save rotation and translation loss of that sequence
		fig_r,ax_r = plt.subplots(1)
		ax_r.plot(avgR_Loss_seq,'r')
		plt.ylabel("Rotation Loss")
		plt.xlabel("Per pair of frames")
		fig_r.savefig(os.path.join(expDir, 'plots', 'loss', 'seq_' + str(seq) + '_rotLoss_valid_epoch_' + str(epoch)))
		
		fig_t,ax_t = plt.subplots(1)
		ax_t.plot(avgT_Loss_seq,'g')
		plt.ylabel("Translation Loss")
		plt.xlabel("Per pair of frames")
		fig_t.savefig(os.path.join(expDir, 'plots', 'loss', 'seq_' + str(seq) + '_trans_Loss_valid_epoch_' + str(epoch)))
		
		fig_tot, ax_tot = plt.subplots(1)
		ax_tot.plot(avgTotal_Loss_seq, 'b')
		plt.ylabel('Total Loss')
		plt.xlabel('Per pair of frames')
		fig_tot.savefig(os.path.join(expDir, 'plots', 'loss', 'seq_' + str(seq) + '_total_Loss_valid_epoch_' + str(epoch)))


		avgRotLoss.append(np.mean(avgR_Loss_seq))
		avgTrLoss.append(np.mean(avgT_Loss_seq))
		avgTotalLoss.append(np.mean(avgTotal_Loss_seq))


	# Return average rotation and translation loss for all the validation sequences.
	return avgRotLoss, avgTrLoss, avgTotalLoss



########################################################################
### Model Definition + Weight init + FlowNet weight loading ###
########################################################################

# Get the definition of the model
if cmd.modelType !="checkpoint_wb":
# Model definition without batchnorm
	deepVO = model.Net_DeepVO_WOB(activation = cmd.activation)

else:

	# Model definition with batchnorm
	deepVO = model.Net_DeepVO_WB()

# Initialize weights (Xavier)
deepVO.init_weights()

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

	# # For the linear layers of the model
	# if cmd.initType == "xavier":
	# 	deepVO.init_weights()

	deepVO.cuda()
	print(' Loaded weights !!')


########################################################################
### Criterion and optimizer ###
########################################################################

criterion = nn.MSELoss()

if cmd.optMethod == "adam":
	optimizer = optim.Adam(deepVO.parameters(), lr = cmd.lr, betas = (cmd.beta1, cmd.beta2), weight_decay = cmd.weightDecay, amsgrad = False)
elif cmd.optMethod == "sgd":
	optimizer = optim.SGD(deepVO.parameters(), lr = cmd.lr, momentum = cmd.momentum, weight_decay = cmd.weightDecay, nesterov = False)
else:
	optimizer = optim.Adagrad(deepVO.parameters(), lr = cmd.lr, lr_decay = cmd.lrDecay , weight_decay = cmd.weightDecay)


########################################################################
###  Main loop ###
########################################################################
r_tr = []
t_tr = []
totalLoss_train = []

r_val=[]
t_val=[]
totalLoss_val = []

for epoch in range(cmd.nepochs):

	print('================> Starting epoch: '  + str(epoch+1) + '/' + str(cmd.nepochs))
	
	# Average loss over one training epoch	
	r_trLoss , t_trLoss, total_trLoss = train(epoch+1)
	r_tr.append(r_trLoss)
	t_tr.append(t_trLoss)
	totalLoss_train.append(total_trLoss)

	print('==> Validation (epoch: ' + str(epoch+1) + ')')
	# Average loss over entire validation set, a list of loss for all sequences
	r_valLoss, t_valLoss, total_valLoss = validate(epoch+1, 'valid')
	r_val.append(np.mean(r_valLoss))
	t_val.append(np.mean(t_valLoss))
	totalLoss_val.append(np.mean(total_valLoss))

	# After all the epochs plot the translation and rotation loss  w.r.t. epochs
	fig_r,ax_r = plt.subplots(1)
	ax_r.plot(r_valLoss)
	plt.ylabel('Rotation Loss : validation')
	plt.xlabel('Per sequence')
	fig_r.savefig(os.path.join(expDir, 'val_rotLoss' + str(epoch+1)))
	
	fig_t,ax_t = plt.subplots(1)
	ax_t.plot(t_valLoss)
	plt.ylabel('Translation Loss : validation')
	plt.xlabel('Per sequence')
	fig_t.savefig(os.path.join(expDir, 'val_transLoss' + str(epoch)))
	
	fig_tot, ax_tot = plt.subplots(1)
	ax_tot.plot(total_valLoss)
	plt.ylabel('Total Loss: validation')
	plt.xlabel('Per sequence')
	fig_tot.savefig(os.path.join(expDir, 'val_totalLoss' + str(epoch)))


# After all the epochs plot the translation and rotation loss  w.r.t. epochs
fig_r,ax_r = plt.subplots(1)
ax_r.plot(r_tr, label = 'train')
ax_r.plot(r_val, label = 'valid')
ax_r.legend()
plt.ylabel('Rotation Loss')
plt.xlabel('Per epoch')
fig_r.savefig(os.path.join(expDir, 'rotLoss'))

fig_t,ax_t = plt.subplots(1)
ax_t.plot(t_tr,label = 'train')
ax_t.plot(t_val,label = 'valid')
ax_t.legend()
plt.ylabel('Translation Loss')
plt.xlabel('Per epoch')
fig_t.savefig(os.path.join(expDir, 'transLoss'))

fig_tot, ax_tot = plt.subplots(1)
ax_tot.plot(totalLoss_train, label = 'Train')
ax_tot.plot(totalLoss_val, label = 'Valid')
ax_tot.legend()
plt.ylabel('Total Loss')
plt.xlabel('Per epoch')
fig_tot.savefig(os.path.join(expDir, 'totalLoss'))

print('Done !!')
