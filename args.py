import argparse
parser = argparse.ArgumentParser()

# Custom FloatRange class, to check for float argument ranges
class FloatRange(object):
	def __init__(self, start, end):
		self.start = start
		self.end = end
	def __eq__(self, other):
		return self.start <= other <= self.end


################ Model Options ################################
parser.add_argument('-loadModel', help = 'Whether or not to load pretrained weights. \
	If yes: then specify the path to the saved weights', default = None)
parser.add_argument('-modelType', help = 'Type of the model to be loaded : \
	1. deepVO |  2. flownet | 3. flownet_batchnorm', type = str.lower, \
	choices = ['deepvo', 'flownet', 'flownet_batchnorm'], default = None)
parser.add_argument('-initType', help = 'Weight initialization for the linear layers', \
	type = str.lower, choices = ['xavier'], default = 'xavier')
parser.add_argument('-activation', help = 'Activation function to be used', type = str.lower, \
	choices = ['relu', 'selu'], default = 'selu')
parser.add_argument('-dropout', help = 'Drop ratio of dropout at penultimate linear layer, \
	if dropout is to be used.', type = float, choices = [FloatRange(0.0, 1.0)])
parser.add_argument('-numLSTMCells', help = 'Number of LSTM cells to stack together', type = int, \
	default = 2)
parser.add_argument('-imageWidth', help = 'Width of the input image', type = int, default = 1280)
parser.add_argument('-imageHeight', help = 'Height of the input image', type = int, default = 384)
parser.add_argument('-freezeCNN', help = 'Whether or not to freeze weights of the CNN', \
	type = bool, default = True)

################ Dataset ######################################
parser.add_argument('-dataset', help = 'dataset to be used for training the network', default = 'KITTI')
parser.add_argument('-outputParameterization', help = 'Parameterization of egomotion to be learnt by the \
	network', type = str.lower, choices = ['default', 'quaternion', 'se3', 'euler', 'mahalanobis'], default = 'default')
parser.add_argument('-outputFrame', help = 'Whether the output is to be in local or in global coordinates', \
	type = str.lower, choices = ['local', 'global'], default = 'local')

################### Hyperparameters ###########################
parser.add_argument('-lr', help = 'Learning rate', type = float, default = 1e-4)
parser.add_argument('-momentum', help = 'Momentum', type = float, default = 0.009)
parser.add_argument('-weightDecay', help = 'Weight decay', type = float, default = 0.)
parser.add_argument('-lrDecay', help = 'Learning rate decay factor', type = float, default = 0.)
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
parser.add_argument('-trainBatch', help = 'train batch size', type = int, default = 40)
parser.add_argument('-validBatch', help = 'valid batch size', type = int, default = 1)

parser.add_argument('-scf', help = 'Scaling factor for the rotation loss terms', \
	type = float, default = 100)
parser.add_argument('-gamma', help = 'For L2 regularization', \
	type = float, default = 0.0)


################### Paths #####################################
parser.add_argument('-cachedir', \
	help = '(Relative path to) directory in which to store logs, models, plots, etc.', \
	type = str, default = 'cache')
parser.add_argument('-datadir', help = 'Absolute path to the directory that holds the dataset', \
	type = str, default = '/data/milatmp1/sharmasa/KITTI/dataset/')

###### Experiments, Snapshots, and Visualization #############
parser.add_argument('-expID', help = 'experiment ID', default = 'tmp')
parser.add_argument('-snapshot', help = 'when to take model snapshots', type = int, default = 5)
parser.add_argument('-snapshotStrategy', help = 'Strategy to save snapshots. Note that this has \
	precedence over the -snapshot argument. 1. none: no snapshot at all | 2. default: as frequently \
	as specified in -snapshot | 3. best: keep only the best performing model thus far, \
	4. recent: keep only the most recent model', type = str.lower, \
	choices = ['none', 'default', 'best', 'recent'])
parser.add_argument('-tensorboardX', help = 'Whether or not to use tensorboardX for \
	visualization', type = bool, default = True)

########### Debugging, Profiling, etc. #######################
parser.add_argument('-debug', help = 'Run in debug mode, and execute 3 quick iterations per train \
	loop. Used in quickly testing whether the code has a silly bug.', type = bool, default = False)
parser.add_argument('-profileGPUUsage', help = 'Profiles GPU memory usage and prints it every \
	train/val batch', type = bool, default = False)
parser.add_argument('-sbatch', help = 'Replaces tqdm and print operations with file writes when \
	True. Useful for reducing I/O when not running in interactive mode (eg. on clusters)', type = bool)

################### Reproducibility ##########################
parser.add_argument('-randomseed', help = 'Seed for pseudorandom number generator', \
	type = int, default = 12345)
parser.add_argument('-isDeterministic', help = 'Whether or not the code should \
	use the provided random seed and run deterministically', type = bool, default = False)
parser.add_argument('-numworkers', help = 'Number of threads available to the DataLoader', \
	type = int, default = 1)


arguments = parser.parse_args()
