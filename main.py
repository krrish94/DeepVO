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
if cmd.modelType != 'checkpoint_wb':
# Model definition without batchnorm
	deepVO = DeepVO(activation = cmd.activation, parameterization = cmd.outputParameterization)
else:
	# Model definition with batchnorm
	deepVO = DeepVO(activation = cmd.activation, parameterization = cmd.outputParameterization, \
		batchnorm = True, flownet_weights_path = cmd.loadModel)

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

for epoch in range(cmd.nepochs):

	print('================> Starting epoch: '  + str(epoch+1) + '/' + str(cmd.nepochs))

	# Create datasets for the current epoch
	kitti_train = KITTIDataset(cmd.datadir, [1], [0], [40])
	kitti_val = KITTIDataset(cmd.datadir, [1], [0], [40])

	# Initialize a trainer (Note that any accumulated gradients on the model are flushed
	# upon creation of this Trainer object)
	trainer = Trainer(cmd, epoch, deepVO, kitti_train, kitti_val, criterion, optimizer, \
		scheduler = None, scaleFactor = cmd.scf)

	# Training loop
	rotLosses_train, transLosses_train, totalLosses_train = trainer.train()

	# Learning rate scheduler, if specified
	if cmd.lrScheduler is not None:
		scheduler.step()

	# Validation loop
	rotLosses_val, transLosses_val, totalLosses_val = trainer.validate()

# # Initialize a quadratic curriculum class
# curriculum = Curriculum(good_loss = 5e-3, min_frames = 10, max_frames = 60, \
# 	curriculum_type = 'quadratic')

# # Number of iterations elapsed (for ease of visualization using tensorboardX)
# iters = 0

# for epoch in range(cmd.nepochs):

# 	print('================> Starting epoch: '  + str(epoch+1) + '/' + str(cmd.nepochs))
	
# 	# Average loss over one training epoch	

# 	r_trLoss , t_trLoss, total_trLoss, iters = train(epoch+1, iters)
# 	r_tr.append(r_trLoss)
# 	t_tr.append(t_trLoss)
# 	totalLoss_train.append(total_trLoss)

# 	print('==> Validation (epoch: ' + str(epoch+1) + ')')
# 	# Average loss over entire validation set, a list of loss for all sequences
# 	r_valLoss, t_valLoss, total_valLoss = validate(epoch+1, 'valid')
# 	r_val.append(np.mean(r_valLoss))
# 	t_val.append(np.mean(t_valLoss))
# 	totalLoss_val.append(np.mean(total_valLoss))

# 	# Scheduler step, if applicable
# 	if cmd.lrScheduler is not None:
# 		scheduler.step()

# 	# Curriculum step
# 	curriculum.step(total_trLoss)

# 	# tensorboardX visualization
# 	if cmd.tensorboardX is True:
# 		writer.add_scalar('loss/val/rot_loss_val', np.mean(r_valLoss), iters)
# 		writer.add_scalar('loss/val/trans_loss_val', np.mean(t_valLoss), iters)
# 		writer.add_scalar('loss/val/total_loss_val', np.mean(total_valLoss), iters)

print('Done !!')

