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
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm, trange

# Other project files with definitions
import args
from Curriculum import Curriculum
from KITTIDataset import KITTIDataset
from Model import DeepVO
from plotTrajectories import plotSequence
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
	deepVO = DeepVO(activation = cmd.activation, parameterization = cmd.outputParameterization)
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

criterion = nn.MSELoss(reduction = 'sum')

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

for epoch in range(cmd.nepochs):

	print('================> Starting epoch: '  + str(epoch+1) + '/' + str(cmd.nepochs))

	# Create datasets for the current epoch
	train_seq = [0, 1, 2, 8, 9]
	train_startFrames = [0, 0, 0, 0, 0]
	train_endFrames = [4540, 1100, 4660, 4070, 1590]
	val_seq = [3, 4, 5, 6, 7, 10]
	val_startFrames = [0, 0, 0, 0, 0, 0]
	val_endFrames = [800, 270, 2760, 1100, 1100, 1200]
	kitti_train = KITTIDataset(cmd.datadir, train_seq, train_startFrames, train_endFrames)
	kitti_val = KITTIDataset(cmd.datadir, val_seq, val_startFrames, val_endFrames)

	# Initialize a trainer (Note that any accumulated gradients on the model are flushed
	# upon creation of this Trainer object)
	trainer = Trainer(cmd, epoch, deepVO, kitti_train, kitti_val, criterion, optimizer, \
		scheduler = None, scaleFactor = cmd.scf)

	# Training loop
	rotLosses_train_cur, transLosses_train_cur, totalLosses_train_cur = trainer.train()

	rotLosses_train += rotLosses_train_cur
	transLosses_train += transLosses_train_cur
	totalLosses_train += totalLosses_train_cur

	# Learning rate scheduler, if specified
	if cmd.lrScheduler is not None:
		scheduler.step()

	# Snapshot
	if epoch % cmd.snapshot == 0 or epoch == cmd.nepochs - 1:
		print('Saving model after epoch', epoch, '...')
		torch.save(deepVO, os.path.join(cmd.expDir, 'models', 'model' + str(epoch).zfill(3) + '.pt'))

	# Validation loop
	rotLosses_val_cur, transLosses_val_cur, totalLosses_val_cur = trainer.validate()

	rotLosses_val += rotLosses_val_cur
	transLosses_val += transLosses_val_cur
	totalLosses_val += totalLosses_val_cur

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
			plotSequence(cmd.expDir, s, seqLen, traj, cmd.datadir, cmd, epoch)
		i += 1


print('Done !!')

