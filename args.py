import argparse
parser = argparse.ArgumentParser()


################ Model Options ################################
parser.add_argument("-loadModel", help="whether to load pretrained weights or not. If you want to: then give the path ", default ="none")
parser.add_argument("-modelType", help="Type of the model to be loaded : three options : 1. caffe |  2. checkpoint_wob | 3. checkpoint_wb", default="none")
parser.add_argument("-initType", help = " weight initialization for the linear layers :: three options : 1. no change | 2. xavier" , default ="noe")

################ Dataset ######################################
parser.add_argument("-dataset",help="dataset to be used for training the network", default="KITTI")

################### Hyperparameters ###########################
parser.add_argument("-lr", help="learning rate",type = float, default = 1e-4)
parser.add_argument("-momentum", help="momentum", type = float, default = 0.009)
parser.add_argument("-weightDecay", help ="weight decay", type = float, default = 0.004)
parser.add_argument("-lrDecay", help = "learning rate decay", type = float, default = 0.5)
parser.add_argument("-iterations", help="number of iterations after which to compute loss", type=int, default =100)

parser.add_argument("-crit", help="criterion type", default="MSE")
parser.add_argument("-optMethod", help="optimization methods : adam | sgd | adagrad ", default="adam")

parser.add_argument("-nepochs", help="number of epochs",type = int, default=50)
parser.add_argument("-trainBatch", help="train batch size", type = int, default =1)
parser.add_argument("-validBatch", help="valid batch size", type = int, default=1)

parser.add_argument("-scf", help = 'scaling factor to reduce the translation loss', type = float, default=0.01)


################### Paths #####################################
parser.add_argument('-cachedir', \
	help = '(Relative path to) directory in which to store logs, models, plots, etc.', \
	type = str, default = 'cache')
parser.add_argument('-datadir', help = 'Absolute path to the directory that holds the dataset', \
	type = str, default = '/data/milatmp1/sharmasa/KITTI/dataset/')


################### Model and experiment #####################
parser.add_argument("-snapshot", help = 'when to take model snapshots', type =int,default = 5)
parser.add_argument("-expID", help = 'experiment ID', default = 'tmp')

parser.add_argument('-randomseed', help = 'Seed for pseudorandom number generator', \
	type = int, default = 12345)



arguments = parser.parse_args()