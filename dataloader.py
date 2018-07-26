from liealgebra import rotMat_to_axisAngle
import lieFunctions
from numpy.linalg import inv
# from pyquaternion import Quaternion
from skimage import io, transform
from skimage.transform import resize

import numpy as np
import os
import random as rn
import scipy.misc as smc
import torch


# Dataloader class. Loader for image sequences and corresponding ground-truth labels.
class Dataloader:

	def __init__(self, datadir, parameterization = 'default'):
		
		# # KITTI dataloader parameters
		# self.r_KITTImean = 88.61
		# self.g_KITTImean = 93.70
		# self.b_KITTImean = 92.11
		
		# self.r_KITTIstddev = 79.35914872
		# self.g_KITTIstddev = 80.69872125
		# self.b_KITTIstddev = 82.34685558

		self.r_KITTImean = 0.0
		self.g_KITTImean = 0.0
		self.b_KITTImean = 0.0
		self.r_KITTIstddev = 1.0
		self.g_KITTIstddev = 1.0
		self.b_KITTIstddev = 1.0

		# 4541, 1101, 4661, 4071, 1591
		#self.train_seqs_KITTI = [0,1,2,8,9]
		#self.test_seqs_KITTI = [3,4,5,6,7,10]
		#self.total_seqs_KITTI =[0,1,2,3,4,5,6,7,8,9,10]

		self.train_seqs_KITTI = [1]
		self.test_seqs_KITTI = [1]
		self.total_seqs_KITTI =[1]

		self.minFrame_KITTI = 2
		self.maxFrame_KITTI = 1095

		# Dimensions to be fed in the input
		self.width_KITTI = 1280
		self.height_KITTI = 384
		self.channels_KITTI = 3

		# Path to KITTI dataset dir
		self.datadir = datadir

		# Output parameterization
		self.parameterization = parameterization


	# Sample start and end indices of a subsequence, given the number of frames to be sampled
	# Ensure that the sampled ranges are valid, i.e., they have valid image pairs and labels.
	def getSubsequence(self, seq, tl, dataset):

		if dataset == 'KITTI':
			seqLength = len(os.listdir(os.path.join(self.datadir, 'sequences', str(seq).zfill(2), \
				'image_2')))

		st_frm = rn.randint(0, seqLength-tl)
		end_frm = st_frm + tl - 1;
		return st_frm, end_frm


	# Center and scale the image, resize and perform other preprocessing tasks
	def preprocessImg(self, img):

		# Subtract the mean R,G,B pixels
		img[:,:,0] = (img[:,:,0] - self.r_KITTImean)/(self.r_KITTIstddev)
		img[:,:,1] = (img[:,:,1] - self.g_KITTImean)/(self.g_KITTIstddev)
		img[:,:,2] = (img[:,:,2] - self.b_KITTImean)/(self.b_KITTIstddev)
	
		# Resize to the dimensions required 
		img = np.resize(img, (self.height_KITTI, self.width_KITTI, self.channels_KITTI))

		# Torch expects NCWH
		img = torch.from_numpy(img)
		img = img.permute(2,0,1)

		return img	


	# Get the image pair and their corresponding R and T.
	def getPairFrameInfo(self, frame1, frame2, seq, dataset):

		if dataset == 'KITTI':

			# Load the two images : loaded as H x W x 3 (R,G,B)
			img1 = smc.imread(os.path.join(self.datadir, 'sequences', str(seq).zfill(2), \
				'image_2', str(frame1).zfill(6) + '.png'), mode = 'RGB')
			img2 = smc.imread(os.path.join(self.datadir, 'sequences', str(seq).zfill(2), \
				'image_2', str(frame2).zfill(6) + '.png'), mode = 'RGB')

			# Preprocess : returned after mean subtraction, resize and NCWH
			img1 = self.preprocessImg(img1)
			img2 = self.preprocessImg(img2)

			pair = torch.empty([1, 2*self.channels_KITTI, self.height_KITTI, self.width_KITTI])
			
			pair[0] = torch.cat((img1, img2), 0)
			
			inputTensor = (pair.float()).cuda()
			inputTensor = inputTensor * torch.from_numpy(np.asarray([1. / 255.], dtype = np.float32)).cuda()
			
			# # Load the poses. The frames are  0 based indexed.
			# poses_old = open(os.path.join(self.datadir, 'poses', str(seq).zfill(2) + '.txt')).readlines()
			# pose_frame1_old = np.concatenate( (np.asarray([(np.float64(i)) for i in poses_old[frame1].split(' ')]).reshape(3,4) , [[0.0,0.0,0.0,1.0]] ), axis=0); # 4x4 transformation matrix
			# pose_frame2_old = np.concatenate( (np.asarray([(np.float64(i)) for i in poses_old[frame2].split(' ')]).reshape(3,4) , [[0.0,0.0,0.0,1.0]] ), axis=0); # 4x4 transformation matrix

			poses = np.loadtxt(os.path.join(self.datadir, 'poses', str(seq).zfill(2) + '.txt'), dtype = np.float32)
			pose_frame1 = np.vstack([np.reshape(poses[frame1].astype(np.float32), (3, 4)), [[0., 0., 0., 1.]]])
			pose_frame2 = np.vstack([np.reshape(poses[frame2].astype(np.float32), (3, 4)), [[0., 0., 0., 1.]]])

			pose2_wrt1 = np.dot(np.linalg.inv(pose_frame1), pose_frame2) # Take a point in frame2 to a point in frame 1 ==> point_1 = (pose2_wrt1)*(point_2)

			if self.parameterization == 'default':
				
				# Extract R and t, convert it to axis angle form
				R = pose2_wrt1[0:3,0:3]
				axisAngle = (torch.from_numpy(np.asarray(rotMat_to_axisAngle(R))).view(-1,3)).float().cuda()
				t = (torch.from_numpy(pose2_wrt1[0:3,3]).view(-1,3)).float().cuda()

				return inputTensor, axisAngle, t

			elif self.parameterization == 'se3':

				# Return a 6-vector of SE(3) exponential coordinates
				pass

			elif self.parameterization == 'quaternion':

				R = pose2_wrt1[0:3, 0:3]
				quat = np.asarray(lieFunctions.rotMat_to_quat(R)).reshape((1,4))
				quaternion = (torch.from_numpy(quat).view(-1,4)).float().cuda()
				t = (torch.from_numpy(pose2_wrt1[0:3,3]).view(-1,3)).float().cuda()

				return inputTensor, quaternion, t

			elif self.parameterization == 'euler':

				R = pose2_wrt1[0:3, 0:3]
				rx, ry, rz = lieFunctions.rotMat_to_euler(R, seq = 'xyz')
				euler = (torch.FloatTensor([rx, ry, rz]).view(-1,3)).cuda()
				t = (torch.from_numpy(pose2_wrt1[0:3,3]).view(-1,3)).float().cuda()

				return inputTensor, euler, t
