"""
Main script: Train and test DeepVO on the KITTI odometry benchmark
"""

from __future__ import print_function, division
import itertools
import random as rn

# The following two lines are needed because of the conda version sets
# 'Qt5Agg' as the default version for matplotlib.use(). The interpreter
# throws a warning that says matplotlib.use('Agg') needs to be called
# before importing pyplot. If the warning is ignored, this results in 
# an error and the code crashes while storing plots (after validation).
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
import torch
# from torch.autograd import Variable as V
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm, trange

# Other project files with definitions
import args
from curriculum import Curriculum
import dataloader
from drawPlot import plotSequences
import model

# Get the commandline arguements
cmd = args.arguments;

# Seed the RNGs (ensure deterministic outputs)
if cmd.isDeterministic:
	rn.seed(cmd.randomseed)
	np.random.seed(cmd.randomseed)
	torch.manual_seed(cmd.randomseed)
	torch.cuda.manual_seed(cmd.randomseed)
	torch.backends.cudnn.deterministic = True

# Debug parameters
if cmd.debug is True:
	debugIters = 3
	cmd.nepochs = 2


# Intitialze the dataloader
dataloader = dataloader.Dataloader(cmd.datadir, parameterization = cmd.outputParameterization)
# Set the default tensor type
torch.set_default_tensor_type(torch.cuda.FloatTensor)

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
### tensorboardX visualization ###
########################################################################

if cmd.tensorboardX is True:
	from tensorboardX import SummaryWriter
	writer = SummaryWriter(log_dir = expDir)


########################################################################
### Train and validation Functions ###
########################################################################

def train(epoch, iters):

	

	#Switching to train mode
	deepVO.train()
	
	trainSeqs = dataloader.train_seqs_KITTI
	# trajLength = list(range(dataloader.minFrame_KITTI, dataloader.maxFrame_KITTI, \
	# 	rn.randint(5, 20)))
	# trajLength = list(itertools.chain.from_iterable(itertools.repeat(x, 100) for x in [20]))
	# trajLength = list(itertools.chain.from_iterable(itertools.repeat(x, 100)) for x in [curriculum.cur_seqlen])
	
	# ???
	trajLength = [curriculum.cur_seqlen for i in range(100)]
	if curriculum.cur_seqlen > curriculum.min_frames:
		trajLength += list(np.random.randint(curriculum.min_frames, curriculum.cur_seqlen, size = 50))

	trajLength = [40 for i in range(50)]	# ???

	rn.shuffle(trainSeqs)
	rn.shuffle(trajLength)

	itt_T_Loss = 0.0
	itt_R_Loss = 0.0
	itt_tot_Loss = 0.0

	num_itt = 0
	
	avgRotLoss = []
	avgTrLoss = []
	avgTotalLoss = []
	
	if cmd.debug:
		numTraj = len(trajLength)
		if numTraj > debugIters:
			trajLength = trajLength[0:debugIters]
		else:
			firstElement = trajLength[0]
			trajLength = list(itertools.chain.from_iterable(itertools.repeat(x, debugIters) \
				for x in [firstElement]))

	for seq in trainSeqs:
		# for tl in trajLength:
		for tl in tqdm(trajLength, unit = 'seqs'):
			# get a random subsequence from 'seq' of length 'fl' : starting index, ending index
			stFrm, enFrm = dataloader.getSubsequence(seq, tl, cmd.dataset)
			# itterate over this subsequence and get the frame data.
			reset_hidden = True

			tqdm.write('Epoch: ' + str(epoch) + ' Sequence : ' + str(seq) + ' Start frame : ' \
				+ str(stFrm) + ' End frame : ' + str(enFrm), file = sys.stdout)

			deepVO.zero_grad()

			loss_r = torch.zeros(1, dtype = torch.float32).cuda()
			loss_t = torch.zeros(1, dtype = torch.float32).cuda()
			loss = torch.zeros(1, dtype = torch.float32).cuda()
			for frm1 in range(stFrm, enFrm):

				inp, axis, t = dataloader.getPairFrameInfo(frm1, frm1+1, seq, cmd.dataset)
				
				# Forward, compute loss and backprop
				# deepVO.zero_grad()					
				output_r, output_t = deepVO.forward(inp, reset_hidden)
				
				batchsize_scale = torch.from_numpy(np.asarray([1. / tl], dtype = np.float32)).cuda()
				loss_r += batchsize_scale * cmd.scf * criterion(output_r, axis)
				loss_t += batchsize_scale * criterion(output_t,t)
				# loss = (1. / cmd.scf) * loss_t
				loss += sum([loss_r, loss_t])
				
				# Soln 1: This works ###
				# loss_r.backward(retain_graph = True)
				# if frm1 != enFrm-1:
				# 	loss_t.backward(retain_graph = True)
				# else:
				# 	loss_t.backward(retain_graph = False)
				
				# Soln 2:
				# loss += sum([loss_r, loss_t])
				# loss.backward()
				# if frm1 != enFrm-1:
				# 	loss.backward(retain_graph = True)
				# else:
				# 	loss.backward(retain_graph = False)

				# Perform gradient clipping
				if cmd.gradClip is not None:
					torch.nn.utils.clip_grad_norm_(deepVO.parameters(), cmd.gradClip)

				# Update model parameters
				# optimizer.step()

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

				reset_hidden = False
				num_itt = num_itt + 1

				if num_itt == cmd.iterations:
					avgRotLoss.append(itt_R_Loss)
					avgTrLoss.append(itt_T_Loss)
					avgTotalLoss.append(itt_tot_Loss)
					num_itt = 0
					itt_T_Loss = 0.0
					itt_R_Loss = 0.0
					itt_tot_Loss = 0.0

				iters += 1



			# Regularization for network weights
			l2_reg = None
			for W in deepVO.parameters():
				if l2_reg is None:
					l2_reg = W.norm(2)
				else:
					l2_reg = l2_reg + W.norm(2)

			l2_reg = cmd.l * l2_reg
			loss = sum([l2_reg,loss])
				
			loss.backward()
					
			# Take optimizer steps.
			optimizer.step()

			# Detach LSTM hidden states
			deepVO.detach_LSTM_hidden()


			tqdm.write('Rot Loss: ' + str(itt_R_Loss) + ' Trans Loss: ' + str(itt_T_Loss), 
				file = sys.stdout)
			tqdm.write('Total Loss: ' + str(itt_tot_Loss), file = sys.stdout)

			# For tensorboardX visualization
			if cmd.tensorboardX is True:
				writer.add_scalar('loss/train/rot_loss_train', itt_R_Loss, iters)
				writer.add_scalar('loss/train/trans_loss_train', itt_T_Loss, iters)
				writer.add_scalar('loss/train/total_loss_train', itt_tot_Loss, iters)
				# writer.add_scalars('loss/train', {'rot_loss_train': itt_R_Loss, \
				# 	'trans_loss_train': itt_T_Loss,	'total_loss_train': itt_tot_Loss}, iters)

					
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
		return 0.0, 0.0, 0.0, iters
	return sum(avgRotLoss)/float(len(avgRotLoss)), sum(avgTrLoss)/float(len(avgTrLoss)), \
	sum(avgTotalLoss)/float(len(avgTotalLoss)), iters


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

	if cmd.debug:
		validSeqs = [validSeqs[0]]

	for seq in tqdm(validSeqs, unit = 'sequences'):

		seqLength = len(os.listdir(os.path.join(cmd.datadir, 'sequences', str(seq).zfill(2), 'image_2')))
		
		# To store the entire estimated trajectory
		if cmd.outputParameterization == 'default' or cmd.outputParameterization == 'euler' or \
		cmd.outputParameterization == 'se3': 
			seq_traj = np.zeros([seqLength-1,6])
		elif cmd.outputParameterization == 'quaternion':
			seq_traj = np.zeros([seqLength-1,7])

		# Loss for each sequence
		avgR_Loss_seq = []
		avgT_Loss_seq = []
		avgTotal_Loss_seq = []

		if cmd.debug:
			seqLength = debugIters
				
		reset_hidden = True

		# for frame1 in trange(seqLength-1):
		for frame1 in trange(39):	# ???

			inp, axis,t = dataloader.getPairFrameInfo(frame1, frame1+1, seq,cmd.dataset)
			
			output_r, output_t = deepVO.forward(inp, reset_hidden)

			# Outputs come in form of torch tensor. Convert to numpy.
			seq_traj[frame1] = np.concatenate((output_r.data.cpu().numpy(), output_t.data.cpu().numpy()), axis = 1)

			loss_r = criterion(output_r,axis)
			loss_t = cmd.scf * criterion(output_t,t)
			loss = sum([loss_r, loss_t])
			
			avgR_Loss_seq.append(loss_r.item())
			avgT_Loss_seq.append(loss_t.item())
			avgTotal_Loss_seq.append(loss.item())
			
			reset_hidden = False

			deepVO.detach_LSTM_hidden()


		tqdm.write('Rot Loss: ' + str(np.mean(avgR_Loss_seq)) + ' Trans Loss: ' + \
			str(np.mean(avgT_Loss_seq)), file = sys.stdout)
		tqdm.write('Total Loss: ' + str(np.mean(avgTotal_Loss_seq)), file = sys.stdout)

		# Plot the trajectory of that sequence
		if tag == "valid":
			plotSequences(expDir, seq, seqLength, seq_traj, cmd.dataset, cmd)


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
	deepVO = model.Net_DeepVO_WOB(activation = cmd.activation, \
		parameterization = cmd.outputParameterization)

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

	deepVO = model.copyWeights(deepVO, flownetModel, cmd.modelType)

	# # For the linear layers of the model
	# if cmd.initType == "xavier":
	# 	deepVO.init_weights()

	deepVO.cuda()
	print(' Loaded weights !!')


# # Helper function to print the norm of the gradient
# def printgradnorm(self, grad_input, grad_output):

# 	print('Inside ' + self.__class__.__name__ + ' backward')
# 	print('grad_input_norm:', grad_input[0].norm().item())
# deepVO.LSTM2.register_backward_hook(printgradnorm)


########################################################################
### Criterion, optimizer, and scheduler ###
########################################################################

criterion = nn.MSELoss(size_average = False)

if cmd.optMethod == 'adam':
	optimizer = optim.Adam(deepVO.parameters(), lr = cmd.lr, betas = (cmd.beta1, cmd.beta2), weight_decay = cmd.weightDecay, amsgrad = False)
elif cmd.optMethod == 'sgd':
	optimizer = optim.SGD(deepVO.parameters(), lr = cmd.lr, momentum = cmd.momentum, weight_decay = cmd.weightDecay, nesterov = False)
else:
	optimizer = optim.Adagrad(deepVO.parameters(), lr = cmd.lr, lr_decay = cmd.lrDecay , weight_decay = cmd.weightDecay)

# Initialize scheduler, if specified
if cmd.lrScheduler is not None:
	if cmd.lrScheduler == 'cosine':
		scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = cmd.nepochs)
	elif cmd.lrScheduler == 'plateau':
		scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

########################################################################
###  Main loop ###
########################################################################
r_tr = []
t_tr = []
totalLoss_train = []

r_val=[]
t_val=[]
totalLoss_val = []

# Initialize a quadratic curriculum class
curriculum = Curriculum(good_loss = 5e-3, min_frames = 10, max_frames = 60, \
	curriculum_type = 'quadratic')

# Number of iterations elapsed
iters = 0

for epoch in range(cmd.nepochs):

	print('================> Starting epoch: '  + str(epoch+1) + '/' + str(cmd.nepochs))
	
	# Average loss over one training epoch	

	r_trLoss , t_trLoss, total_trLoss, iters = train(epoch+1, iters)
	r_tr.append(r_trLoss)
	t_tr.append(t_trLoss)
	totalLoss_train.append(total_trLoss)

	print('==> Validation (epoch: ' + str(epoch+1) + ')')
	# Average loss over entire validation set, a list of loss for all sequences
	r_valLoss, t_valLoss, total_valLoss = validate(epoch+1, 'valid')
	r_val.append(np.mean(r_valLoss))
	t_val.append(np.mean(t_valLoss))
	totalLoss_val.append(np.mean(total_valLoss))

	# Scheduler step, if applicable
	if cmd.lrScheduler is not None:
		scheduler.step()

	# Curriculum step
	curriculum.step(total_trLoss)

	# tensorboardX visualization
	if cmd.tensorboardX is True:
		writer.add_scalar('loss/val/rot_loss_val', np.mean(r_valLoss), iters)
		writer.add_scalar('loss/val/trans_loss_val', np.mean(t_valLoss), iters)
		writer.add_scalar('loss/val/total_loss_val', np.mean(total_valLoss), iters)
		# writer.add_scalars('loss/val', {'rot_loss_val': np.mean(r_valLoss), \
		# 	'trans_loss_val': np.mean(t_valLoss), 'total_loss_val': np.mean(total_valLoss)}, iters)

	# After all the epochs plot the translation and rotation loss  w.r.t. epochs
	fig_r,ax_r = plt.subplots(1)
	ax_r.plot(r_val)
	plt.ylabel('Rotation Loss : validation')
	plt.xlabel('Per sequence')
	fig_r.savefig(os.path.join(expDir, 'val_rotLoss' + str(epoch+1)))
	
	fig_t,ax_t = plt.subplots(1)
	ax_t.plot(t_val)
	plt.ylabel('Translation Loss : validation')
	plt.xlabel('Per sequence')
	fig_t.savefig(os.path.join(expDir, 'val_transLoss' + str(epoch)))
	
	fig_tot, ax_tot = plt.subplots(1)
	ax_tot.plot(totalLoss_val)
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
