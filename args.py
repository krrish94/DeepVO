import argparse
parser = argparse.ArgumentParser()


################ Model Options ################################
parser.add_argument("-loadModel", help="whether to load pretrained weights or not. If you want to: then give the path ", default ="none")
parser.add_argument("-modelType", help="Type of the model to be loaded : three options : 1. caffe |  2. checkpoint_wob | 3. checkpoint_wb", default="none")

################ Dataset ######################################
parser.add_argument("-dataset",help="dataset to be used for traingin the network", default="KITTI")

################### Hyperparameters ###########################
parser.add_argument("-lr", help="learning rate",type = float, default = 2.5e-5)
parser.add_argument("-momentum", help="momentum", type = float, default = 0.009)
parser.add_argument("-weightDecay", help ="weight decay", type = float, default = 0.004)
parser.add_argument("-lrDecay", help ="learning rate decay", type = float, default = 0.5)

parser.add_argument("-crit", help="criterion type", default="MSE")
parser.add_argument("-optMethod", help="optimization methods : adam | sgd | adagrad ", default="adam")

parser.add_argument("-nepochs", help="number of epochs",type = int, default=2)
parser.add_argument("-trainBatch", help="train batch size", type = int, default =1)
parser.add_argument("-validBatch", help="valid batch size", type = int, default=1)


################### Snapshot of the model #####################






arguments = parser.parse_args()