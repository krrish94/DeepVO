"""
Main script: Train and test DeepVO on the KITTI odometry benchmark
"""

# The following two lines are needed because, conda on Mila SLURM sets
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
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

# Other project files with definitions
import args
from KITTIDataset import KITTIDataset
from losses import MahalanobisLoss
from Model import DeepVO
from plotTrajectories import plotSequenceRelative, plotSequenceAbsolute
from Trainer import Trainer


# Parse commandline arguements
cmd = args.arguments;

# Seed the RNGs (ensure deterministic outputs), if specified via commandline
if cmd.isDeterministic:
	# rn.seed(cmd.randomseed)
	np.random.seed(cmd.randomseed)
	torch.manual_seed(cmd.randomseed)
	torch.cuda.manual_seed(cmd.randomseed)
	torch.backends.cudnn.deterministic = True


# Debug parameters. This is to run in 'debug' mode, which runs a very quick pass
# through major segments of code, to ensure nothing awful happens when we deploy
# on GPU clusters for instance, as a batch script. It is sometimes very annoying 
# when code crashes after a few epochs of training, while attempting to write a 
# checkpoint to a directory that does not exist.
if cmd.debug is True:
	cmd.debugIters = 3
	cmd.nepochs = 2

# Set default tensor type to cuda.FloatTensor, for GPU execution
torch.set_default_tensor_type(torch.cuda.FloatTensor)

# Create directory structure, to store results
cmd.basedir = os.path.dirname(os.path.realpath(__file__))
if not os.path.exists(os.path.join(cmd.basedir, cmd.cachedir, cmd.dataset)):
	os.makedirs(os.path.join(cmd.basedir, cmd.cachedir, cmd.dataset))

cmd.expDir = os.path.join(cmd.basedir, cmd.cachedir, cmd.dataset, cmd.expID)
if not os.path.exists(cmd.expDir):
	os.makedirs(cmd.expDir)
	print('Created dir: ', cmd.expDir)
if not os.path.exists(os.path.join(cmd.expDir, 'models')):
	os.makedirs(os.path.join(cmd.expDir, 'models'))
	print('Created dir: ', os.path.join(cmd.expDir, 'models'))
if not os.path.exists(os.path.join(cmd.expDir, 'plots', 'traj')):
	os.makedirs(os.path.join(cmd.expDir, 'plots', 'traj'))
	print('Created dir: ', os.path.join(cmd.expDir, 'plots', 'traj'))
if not os.path.exists(os.path.join(cmd.expDir, 'plots', 'loss')):
	os.makedirs(os.path.join(cmd.expDir, 'plots', 'loss'))
	print('Created dir: ', os.path.join(cmd.expDir, 'plots', 'loss'))
for seq in range(11):
	if not os.path.exists(os.path.join(cmd.expDir, 'plots', 'traj', str(seq).zfill(2))):
		os.makedirs(os.path.join(cmd.expDir, 'plots', 'traj', str(seq).zfill(2)))
		print('Created dir: ', os.path.join(cmd.expDir, 'plots', 'traj', str(seq).zfill(2)))

# Save all the command line arguements in a text file in the experiment directory.
cmdFile = open(os.path.join(cmd.expDir, 'args.txt'), 'w')
for arg in vars(cmd):
	cmdFile.write(arg + ' ' + str(getattr(cmd, arg)) + '\n')
cmdFile.close()

# TensorboardX visualization support
if cmd.tensorboardX is True:
	from tensorboardX import SummaryWriter
	writer = SummaryWriter(log_dir = cmd.expDir)


########################################################################
### Model Definition + Weight init + FlowNet weight loading ###
########################################################################

# Get the definition of the model
if cmd.modelType == 'flownet' or cmd.modelType is None:
	# Model definition without batchnorm
	deepVO = DeepVO(cmd.imageWidth, cmd.imageHeight, activation = cmd.activation, parameterization = cmd.outputParameterization, \
		numLSTMCells = cmd.numLSTMCells, hidden_units_LSTM = [1024, 1024])
elif cmd.modelType == 'flownet_batchnorm':
	# Model definition with batchnorm
	deepVO = DeepVO(activation = cmd.activation, parameterization = cmd.outputParameterization, \
		batchnorm = True, flownet_weights_path = cmd.loadModel)

# Load a pretrained DeepVO model
if cmd.modelType == 'deepvo':
	deepVO = torch.load(cmd.loadModel)
else:
	# Initialize weights for fully connected layers and for LSTMCells
	deepVO.init_weights()
	# CUDAfy
	deepVO.cuda()
print('Loaded! Good to launch!')


########################################################################
### Criterion, optimizer, and scheduler ###
########################################################################

if cmd.outputParameterization == 'mahalanobis':
	criterion = MahalanobisLoss
else:
	criterion = nn.MSELoss(reduction = 'sum')

if cmd.freezeCNN is True:
	n = 0
	for p in deepVO.parameters():
		if p.requires_grad is True:
			# The first 18 trainable parameters correspond to the CNN (FlowNetS)
			if n <= 17:
				p.requires_grad = False
				n += 1
	# print(len(list(filter(lambda p: p.requires_grad, deepVO.parameters()))))
	# sys.exit(1)

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

rotLosses_train = []
transLosses_train = []
totalLosses_train = []
rotLosses_val = []
transLosses_val = []
totalLosses_val = []
bestValLoss = np.inf


# Create datasets for the current epoch
train_seq = [0, 1, 2, 8, 9]
train_startFrames = [0, 0, 0, 0, 0]
train_endFrames = [4540, 1100, 4660, 4070, 1590]
val_seq = [3, 4, 5, 6, 7, 10]
val_startFrames = [0, 0, 0, 0, 0, 0]
val_endFrames = [800, 270, 2760, 1100, 1100, 1200]
# train_seq = [0]
# train_startFrames = [40]
# train_endFrames = [240]
# val_seq = [0]
# val_startFrames = [40]
# val_endFrames = [240]


for epoch in range(cmd.nepochs):

	print('================> Starting epoch: '  + str(epoch+1) + '/' + str(cmd.nepochs))

	train_seq_cur_epoch = []
	train_startFrames_cur_epoch = []
	train_endFrames_cur_epoch = []
	# Take each sequence and split it into chunks
	for s in range(len(train_seq)):
		MAX_NUM_CHUNKS = 1
		num_chunks = 0
		if (train_endFrames[s] - train_startFrames[s]) // cmd.trainBatch != 0:
			num_chunks = np.random.randint(0, min((MAX_NUM_CHUNKS, (train_endFrames[s] - train_startFrames[s]) // cmd.trainBatch)))
		# We don't need no chunks. We need at least one
		if num_chunks == 0:
			num_chunks = 1
		cur_seq = [idx for idx in range(train_startFrames[s], train_endFrames[s], (train_endFrames[s] - train_startFrames[s]) // num_chunks)]
		for j in range(len(cur_seq)-1):
			train_seq_cur_epoch.append(train_seq[s])
			train_startFrames_cur_epoch.append(cur_seq[j])
			train_endFrames_cur_epoch.append(cur_seq[j+1]-1)
		if len(cur_seq) == 1: # Corner case
			train_seq_cur_epoch.append(train_seq[s])
			train_startFrames_cur_epoch.append(train_startFrames[s])
			train_endFrames_cur_epoch.append(train_endFrames[s])
	permutation = np.random.permutation(len(train_seq_cur_epoch))
	train_seq_cur_epoch = [train_seq_cur_epoch[p] for p in permutation]
	train_startFrames_cur_epoch = [train_startFrames_cur_epoch[p] for p in permutation]
	train_endFrames_cur_epoch = [train_endFrames_cur_epoch[p] for p in permutation]

	kitti_train = KITTIDataset(cmd.datadir, train_seq_cur_epoch, train_startFrames_cur_epoch, \
		train_endFrames_cur_epoch, width = cmd.imageWidth, height = cmd.imageHeight, \
		parameterization = cmd.outputParameterization, outputFrame = cmd.outputFrame)
	kitti_val = KITTIDataset(cmd.datadir, val_seq, val_startFrames, val_endFrames, \
		width = cmd.imageWidth, height = cmd.imageHeight, parameterization = cmd.outputParameterization, \
		outputFrame = cmd.outputFrame)

	# dataloader_train = DataLoader(kitti_train, batch_size = 1, shuffle = False, \
	# 	num_workers = cmd.numWorkers)
	# dataloader_val = DataLoader(kitti_val, batch_size = 1, shuffle = False, \
	# 	num_workers = cmd.numWorkers)

	# Initialize a trainer (Note that any accumulated gradients on the model are flushed
	# upon creation of this Trainer object)
	trainer = Trainer(cmd, epoch, deepVO, kitti_train, kitti_val, criterion, optimizer, \
		scheduler = None)

	# Training loop
	print('===> Training: '  + str(epoch+1) + '/' + str(cmd.nepochs))
	startTime = time.time()
	rotLosses_train_cur, transLosses_train_cur, totalLosses_train_cur = trainer.train()
	print('Train time: ', time.time() - startTime)

	rotLosses_train += rotLosses_train_cur
	transLosses_train += transLosses_train_cur
	totalLosses_train += totalLosses_train_cur

	# Learning rate scheduler, if specified
	if cmd.lrScheduler is not None:
		scheduler.step()

	# Snapshot
	if cmd.snapshotStrategy == 'default':
		if epoch % cmd.snapshot == 0 or epoch == cmd.nepochs - 1:
			print('Saving model after epoch', epoch, '...')
			torch.save(deepVO, os.path.join(cmd.expDir, 'models', 'model' + str(epoch).zfill(3) + '.pt'))
	elif cmd.snapshotStrategy == 'recent':
		# Save the most recent model
		print('Saving model after epoch', epoch, '...')
		torch.save(deepVO, os.path.join(cmd.expDir, 'models', 'recent.pt'))
	elif cmd.snapshotStrategy == 'best' or 'none':
		# If we only want to save the best model, defer the decision
		pass

	# Validation loop
	print('===> Validation: '  + str(epoch+1) + '/' + str(cmd.nepochs))
	startTime = time.time()
	rotLosses_val_cur, transLosses_val_cur, totalLosses_val_cur = trainer.validate()
	print('Val time: ', time.time() - startTime)

	rotLosses_val += rotLosses_val_cur
	transLosses_val += transLosses_val_cur
	totalLosses_val += totalLosses_val_cur

	# Snapshot (if using 'best' strategy)
	if cmd.snapshotStrategy == 'best':
		if np.mean(totalLosses_val_cur) <= bestValLoss:
			bestValLoss = np.mean(totalLosses_val_cur)
			print('Saving model after epoch', epoch, '...')
			torch.save(deepVO, os.path.join(cmd.expDir, 'models', 'best' + '.pt'))

	if cmd.tensorboardX is True:
		writer.add_scalar('loss/train/rot_loss_train', np.mean(rotLosses_train), trainer.iters)
		writer.add_scalar('loss/train/trans_loss_train', np.mean(transLosses_train), trainer.iters)
		writer.add_scalar('loss/train/total_loss_train', np.mean(totalLosses_train), trainer.iters)
		writer.add_scalar('loss/train/rot_loss_val', np.mean(rotLosses_val), trainer.iters)
		writer.add_scalar('loss/train/trans_loss_val', np.mean(transLosses_val), trainer.iters)
		writer.add_scalar('loss/train/total_loss_val', np.mean(totalLosses_val), trainer.iters)

	# Save training curves
	fig, ax = plt.subplots(1)
	ax.plot(range(len(rotLosses_train)), rotLosses_train, 'r', label = 'rot_train')
	ax.plot(range(len(transLosses_train)), transLosses_train, 'g', label = 'trans_train')
	ax.plot(range(len(totalLosses_train)), totalLosses_train, 'b', label = 'total_train')
	ax.legend()
	plt.ylabel('Loss')
	plt.xlabel('Batch #')
	fig.savefig(os.path.join(cmd.expDir, 'loss_train_' + str(epoch).zfill(3)))

	fig, ax = plt.subplots(1)
	ax.plot(range(len(rotLosses_val)), rotLosses_val, 'r', label = 'rot_train')
	ax.plot(range(len(transLosses_val)), transLosses_val, 'g', label = 'trans_val')
	ax.plot(range(len(totalLosses_val)), totalLosses_val, 'b', label = 'total_val')
	ax.legend()
	plt.ylabel('Loss')
	plt.xlabel('Batch #')
	fig.savefig(os.path.join(cmd.expDir, 'loss_val_' + str(epoch).zfill(3)))

	# Plot trajectories (validation sequences)
	i = 0
	for s in val_seq:
		seqLen = val_endFrames[i] - val_startFrames[i]
		trajFile = os.path.join(cmd.expDir, 'plots', 'traj', str(s).zfill(2), \
			'traj_' + str(epoch).zfill(3) + '.txt')
		if os.path.exists(trajFile):
			traj = np.loadtxt(trajFile)
			traj = traj[:,3:]
			if cmd.outputFrame == 'local':
				plotSequenceRelative(cmd.expDir, s, seqLen, traj, cmd.datadir, cmd, epoch)
			elif cmd.outputFrame == 'global':
				plotSequenceAbsolute(cmd.expDir, s, seqLen, traj, cmd.datadir, cmd, epoch)
		i += 1


print('Done !!')

