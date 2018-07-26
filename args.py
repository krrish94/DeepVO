import argparse
parser = argparse.ArgumentParser()


################ Model Options ################################
parser.add_argument('-loadModel', help = 'Whether or not to load pretrained weights. \
	If yes: then specify the path to the saved weights', default = 'none')
parser.add_argument('-modelType', help = 'Type of the model to be loaded : \
	1. caffe |  2. checkpoint_wob | 3. checkpoint_wb', type = str.lower, \
	choices = ['caffe', 'checkpoint_wb', 'checkpoint_wob'], default = 'none')
parser.add_argument('-initType', help = 'Weight initialization for the linear layers', \
	type = str.lower, choices = ['xavier'], default = 'xavier')
parser.add_argument('-activation', help = 'Activation function to be used', type = str.lower, \
	choices = ['relu', 'selu'], default = 'selu')

################ Dataset ######################################
parser.add_argument('-dataset', help = 'dataset to be used for training the network', default = 'KITTI')
parser.add_argument('-outputParameterization', help = 'Parameterization of egomotion to be learnt by the \
	network', type = str.lower, choices = ['default', 'quaternion', 'se3'], default = 'default')

################### Hyperparameters ###########################
parser.add_argument('-lr', help = 'Learning rate', type = float, default = 1e-4)
parser.add_argument('-momentum', help = 'Momentum', type = float, default = 0.009)
parser.add_argument('-weightDecay', help = 'Weight decay', type = float, default = 0.004)
parser.add_argument('-lrDecay', help = 'Learning rate decay factor', type = float, default = 0.5)
parser.add_argument('-iterations', help = 'Number of iterations after loss is to be computed', \
	type = int, default = 100)
parser.add_argument('-beta1', help = 'beta1 for ADAM optimizer', type = float, default = 0.9)
parser.add_argument('-beta2', help = 'beta2 for ADAM optimizer', type = float, default = 0.999)
parser.add_argument('-gradClip', help = 'Max allowed magnitude for the gradient norm, \
	if gradient clipping is to be performed. (Recommended: 1.0)', type = float)

# parser.add_argument('-crit', help = 'Error criterion', default = 'MSE')
parser.add_argument('-optMethod', help = 'Optimization method : adam | sgd | adagrad ', \
	type = str.lower, choices = ['adam', 'sgd', 'adagrad'], default = 'adam')
parser.add_argument('-lrScheduler', help = 'Learning rate scheduler', type = str.lower, \
	choices = ['cosine', 'plateau'])

parser.add_argument('-nepochs', help = 'Number of epochs', type = int, default = 50)
parser.add_argument('-trainBatch', help = 'train batch size', type = int, default = 1)
parser.add_argument('-validBatch', help = 'valid batch size', type = int, default = 1)

parser.add_argument('-scf', help = 'Scaling factor for the translation loss terms', \
	type = float, default = 0.01)


################### Paths #####################################
parser.add_argument('-cachedir', \
	help = '(Relative path to) directory in which to store logs, models, plots, etc.', \
	type = str, default = 'cache')
parser.add_argument('-datadir', help = 'Absolute path to the directory that holds the dataset', \
	type = str, default = '/data/milatmp1/sharmasa/KITTI/dataset/')

###### Experiments, Snapshots, and Visualization #############
parser.add_argument('-expID', help = 'experiment ID', default = 'tmp')
parser.add_argument('-snapshot', help = 'when to take model snapshots', type = int, default = 5)
parser.add_argument('-tensorboardX', help = 'Whether or not to use tensorboardX for \
	visualization', type = bool, default = True)
parser.add_argument('-debug', help = 'Run in debug mode, and execute 3 quick iterations per train \
	loop. Used in quickly testing whether the code has a silly bug.', type = bool, default = False)

################### Reproducibility ##########################
parser.add_argument('-randomseed', help = 'Seed for pseudorandom number generator', \
	type = int, default = 12345)
parser.add_argument('-isDeterministic', help = 'Whether or not the code should \
	use the provided random seed and run deterministically', type = bool, default = True)


arguments = parser.parse_args()
