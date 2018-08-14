import helpers
from lieFunctions import rotMat_to_axisAngle, rotMat_to_quat, rotMat_to_euler
import numpy as np
import os
import scipy.misc as smc
from skimage import io
import torch
from torch.utils.data import Dataset

# Class for providing an iterator for the KITTI visual odometry dataset
class KITTIDataset(Dataset):

	# Constructor
	def __init__(self, KITTIBaseDir, sequences = None, startFrames = None, endFrames = None, \
		parameterization = 'default', width = 1280, height = 384, outputFrame = 'local'):

		# Path to base directory of the KITTI odometry dataset
		# The base directory contains two directories: 'poses' and 'sequences'
		# The 'poses' directory contains text files that contain ground-truch pose 
		# for the train sequences (00-10). The 11 train sequences and 11 test sequences
		# are present in the 'sequences' folder
		self.baseDir = KITTIBaseDir

		# Path to directory containing images
		self.imgDir = os.path.join(self.baseDir, 'sequences')
		# Path to directory containing pose ground-truth
		self.poseDir = os.path.join(self.baseDir, 'poses')

		# Max frames in each KITTI sequence
		self.KITTIMaxFrames = [4540, 1100, 4660, 800, 270, 2760, 1100, 1100, 4070, 1590, 1200]

		# # Mean of R, G, B color channel values
		# self.channelwiseMean = [88.61, 93.70, 92.11]
		# # Standard deviation of R, G, B color channel values
		# self.channelwiseStdDev = [79.35914872, 80.69872125, 82.34685558]

		self.channelwiseMean = [0.0, 0.0, 0.0]
		# Got the following values from Clement Pinard, whose PyTorch FlowNet we're using
		self.channelwiseMean = [0.411,0.432,0.45]
		self.channelwiseStdDev = [1.0, 1.0, 1.0]

		# Dimensions to be fed in the input
		self.width = width
		self.height = height
		self.channels = 3

		# List of sequences that are part of the dataset
		# If nothing is specified, use sequence 1 as default
		self.sequences = sequences if sequences is not None else list(1)

		# List of start frames and end frames for each sequence
		self.startFrames = startFrames if startFrames is not None else list(0)
		self.endFrames = endFrames if endFrames is not None else list(1100)

		# Parameterization to be used to represent the transformation
		self.parameterization = parameterization
		self.outputFrame = outputFrame

		if self.parameterization == 'mahalanobis':
			# Covariance matrix for pose estimates (used in mahalanobis distance parameterization)
			self.infoMat = torch.from_numpy(np.asarray([[1.031282396606333059e+05, -1.161143275344443737e+03, -3.025711229425782676e+03, 1.616772034397704871e+01, 7.874578620220295306e+02, 1.933107706768316802e+01], \
				[-1.161143275344419180e+03, 8.652048945332411677e+03, 5.858869701431300200e+03, -4.946301428859676889e+03, 1.029099273494638425e+02, 1.737187519897137733e+01], \
				[-3.025711229425711281e+03, 5.858869701431291105e+03, 1.486145624393749749e+05, -7.933953288062671163e+03, -1.402672240739992446e+02, 3.150564292104001396e+01], \
				[1.616772034395815538e+01, -4.946301428859677799e+03, -7.933953288062682077e+03, 4.465953569539429736e+03, -1.592027587711049250e+02, -1.814574145229429902e+01], \
				[7.874578620221244591e+02, 1.029099273494619524e+02, -1.402672240740190830e+02, -1.592027587711041576e+02, 3.107135710768782701e+03, 4.642915638990846361e+01], \
				[1.933107706768687351e+01, 1.737187519897126009e+01, 3.150564292103979724e+01, -1.814574145229425994e+01, 4.642915638990842098e+01, 5.491082014422132396e+00]])).float().cuda()
		else:
			self.infoMat = None

		# Variable to hold length of the dataset
		self.len = 0
		# Variables used as caches to implement quick __getitem__ retrieves
		self.cumulativeLengths = [0 for i in range(len(self.sequences))]

		# Check if the parameters passed are consistent. Throw an error otherwise
		# KITTI has ground-truth pose information only for sequences 00 to 10
		if min(self.sequences) < 0 or max(self.sequences) > 10:
			raise ValueError('Sequences must be within the range [00-10]')
		if len(self.sequences) != len(self.startFrames):
			raise ValueError('There are not enough startFrames specified as there are sequences.')
		if len(self.sequences) != len(self.endFrames):
			raise ValueError('There are not enough endFrames specified as there are sequences.')
		# Check that, for each sequence, the corresponding start and end frames are within limits
		for i in range(len(self.sequences)):
			seq = self.sequences[i]
			if self.startFrames[i] < 0 or self.startFrames[i] > self.KITTIMaxFrames[seq]:
				raise ValueError('Invalid startFrame for sequence', str(seq).zfill(2))
			if self.endFrames[i] < 0 or self.endFrames[i] <= self.startFrames[i] or \
			self.endFrames[i] > self.KITTIMaxFrames[seq]:
				raise ValueError('Invalid endFrame for sequence', str(seq).zfill(2))
			self.len += (endFrames[i] - startFrames[i])
			self.cumulativeLengths[i] = self.len
		if self.len < 0:
			raise ValueError('Length of the dataset cannot be negative.')


	# Get dataset size
	def __len__(self):

		return self.len


	# __getitem__ method: retrieves an item from the dataset at a specific index
	def __getitem__(self, idx):

		# First determine which sequence the index belongs to, using self.cumulativeLengths
		seqKey = helpers.firstGE(self.cumulativeLengths, idx)
		seqIdx = self.sequences[seqKey]

		# Now select the offset from the first frame of the sequence that the current idx
		# belongs to
		if seqKey == 0:
			offset = idx
		else:
			offset = idx - self.cumulativeLengths[seqKey-1]

		# Map the offset to frame ids
		frame1 = self.startFrames[seqKey] + offset
		frame2 = frame1 + 1

		# Flag to indicate end of sequence
		endOfSequence = False
		if frame2 == self.endFrames[seqKey]:
			endOfSequence = True

		# return (seqIdx, frame1, frame2)

		# Directory containing images for the current sequence
		curImgDir = os.path.join(self.imgDir, str(seqIdx).zfill(2), 'image_2')
		# Read in the corresponding images
		# print(os.path.join(curImgDir, str(frame1).zfill(6) + '.png'))
		# print(os.path.join(curImgDir, str(frame2).zfill(6) + '.png'))
		img1 = smc.imread(os.path.join(curImgDir, str(frame1).zfill(6) + '.png'), mode = 'RGB')
		img2 = smc.imread(os.path.join(curImgDir, str(frame2).zfill(6) + '.png'), mode = 'RGB')
		# Preprocess : returned after mean subtraction, resize and permute to N x C x W x H dims
		img1 = self.preprocessImg(img1)
		img2 = self.preprocessImg(img2)

		# Concatenate the images along the channel dimension (and CUDAfy them)
		pair = torch.empty([1, 2*self.channels, self.height, self.width])	
		pair[0] = torch.cat((img1, img2), 0)
		inputTensor = (pair.float()).cuda()
		inputTensor = inputTensor * torch.from_numpy(np.asarray([1. / 255.], \
			dtype = np.float32)).cuda()

		# Load pose ground-truth
		poses = np.loadtxt(os.path.join(self.poseDir, str(seqIdx).zfill(2) + '.txt'), \
			dtype = np.float32)
		# If using global transformations, load GT transformation of startFrame wrt world
		if self.outputFrame == 'global':
			pose1 = np.vstack([np.reshape(poses[self.startFrames[seqKey]].astype(np.float32), (3, 4)), \
			[[0., 0., 0., 1.]]])
		# Else load GT transformation of frame1 wrt world
		elif self.outputFrame == 'local':
			pose1 = np.vstack([np.reshape(poses[frame1].astype(np.float32), (3, 4)), \
				[[0., 0., 0., 1.]]])
		# Regardless of using local or global transformations, we need to load GT transformation
		# of frame2 wrt world
		pose2 = np.vstack([np.reshape(poses[frame2].astype(np.float32), (3, 4)), \
			[[0., 0., 0., 1.]]])
		# Compute relative pose from frame1/startFrame (local/global) to frame2
		pose2wrt1 = np.dot(np.linalg.inv(pose1), pose2)
		R = pose2wrt1[0:3,0:3]
		t = (torch.from_numpy(pose2wrt1[0:3,3]).view(-1,3)).float().cuda()

		# Default parameterization: representation rotations as axis-angle vectors
		if self.parameterization == 'default' or self.parameterization == 'mahalanobis':
			axisAngle = (torch.from_numpy(np.asarray(rotMat_to_axisAngle(R))).view(-1,3)).float().cuda()
			if self.parameterization == 'default':
				return inputTensor, axisAngle, t, seqIdx, frame1, frame2, endOfSequence
			elif self.parameterization == 'mahalanobis':
				return inputTensor, torch.cat((axisAngle, t), dim = 1), None, seqIdx, frame1, frame2, endOfSequence
		# Quaternion parameterization: representation rotations as quaternions
		elif self.parameterization == 'quaternion':
			quat = np.asarray(rotMat_to_quat(R)).reshape((1,4))
			quaternion = (torch.from_numpy(quat).view(-1,4)).float().cuda()
			return inputTensor, quaternion, t, seqIdx, frame1, frame2, endOfSequence
		# Euler parameterization: representation rotations as Euler angles
		elif self.parameterization == 'euler':
			rx, ry, rz = rotMat_to_euler(R, seq = 'xyz')
			euler = (torch.FloatTensor([rx, ry, rz]).view(-1,3)).cuda()
			return inputTensor, euler, t, seqIdx, frame1, frame2, endOfSequence

		# return (seqIdx, frame1, frame2)


	# Center and scale the image, resize and perform other preprocessing tasks
	def preprocessImg(self, img):

		# Subtract the mean R,G,B pixels
		img[:,:,0] = (img[:,:,0] - self.channelwiseMean[0])/(self.channelwiseStdDev[0])
		img[:,:,1] = (img[:,:,1] - self.channelwiseMean[1])/(self.channelwiseStdDev[1])
		img[:,:,2] = (img[:,:,2] - self.channelwiseMean[2])/(self.channelwiseStdDev[2])
	
		# Resize to the dimensions required 
		img = np.resize(img, (self.height, self.width, self.channels))

		# Torch expects NCWH
		img = torch.from_numpy(img)
		img = img.permute(2,0,1)

		return img
