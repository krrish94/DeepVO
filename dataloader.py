zfrom skimage import io, transform
from skimage.transform import resize
from numpy.linalg import inv
from pyquaternion import Quaternion
from liealgebra import rotMat_to_axisAngle


import numpy as np
import os
import random as rn
import scipy.misc as smc
import torch



class Dataloader:
	def __init__(self):

		
		# KITTI dataloader parameters
		self.r_KITTImean = 88.61
		self.g_KITTImean = 93.70
		self.b_KITTImean = 92.11
		
		self.r_KITTIstddev = 79.35914872;
		self.g_KITTIstddev = 80.69872125;
		self.b_KITTIstddev = 82.34685558;
		
		# 4541, 1101, 4661, 4071, 1591
		#self.train_seqs_KITTI = [0,1,2,8,9]
		#self.test_seqs_KITTI = [3,4,5,6,7,10]
		#self.total_seqs_KITTI =[0,1,2,3,4,5,6,7,8,9,10]

		self.train_seqs_KITTI = [1]
		self.test_seqs_KITTI = [1]
		self.total_seqs_KITTI =[1]

		self.minFrame_KITTI = 2;
		self.maxFrame_KITTI = 1095;

		# Dimensions to be fed in the input
		self.width_KITTI = 1280;
		self.height_KITTI = 384;
		self.channels_KITTI = 3;

	# Get start and end of a subsequence 
	def getSubsequence(self,seq,tl,dataset):
		if dataset == "KITTI":
			seqLength = len(os.listdir("/data/milatmp1/sharmasa/"+ dataset + "/dataset/sequences/" + str(seq).zfill(2) + "/image_2/"))

		st_frm = rn.randint(0, seqLength-tl)
		end_frm = st_frm + tl - 1;
		return st_frm, end_frm

	def preprocessImg(self,img):
		# Subtract the mean R,G,B pixels
		img[:,:,0] = img[:,:,0] - self.r_KITTImean 
		img[:,:,1] = img[:,:,1] - self.g_KITTImean 
		img[:,:,2] = img[:,:,2] - self.b_KITTImean
	
		# Resize to the dimensions required 
		img = np.resize(img,(self.height_KITTI,self.width_KITTI,self.channels_KITTI))

		# Torch expects NCWH
		img = torch.from_numpy(img)
		img = img.permute(2,0,1)

		return img


		


	# Get the image pair and their corresponding R and T.
	def getPairFrameInfo(self,frame1,frame2,seq,dataset):

		if dataset == "KITTI":
			# Load the two images : loaded as H x W x 3(R,G,B)
			img1 = smc.imread("/data/milatmp1/sharmasa/"+ dataset + "/dataset/sequences/" + str(seq).zfill(2) + "/image_2/" + str(frame1).zfill(6) + ".png")
			img2 = smc.imread("/data/milatmp1/sharmasa/"+ dataset + "/dataset/sequences/" + str(seq).zfill(2) + "/image_2/" + str(frame2).zfill(6) + ".png")
			
			# Preprocess : returned after mean subtraction, resize and NCWH
			img1 = self.preprocessImg(img1)
			img2 = self.preprocessImg(img2)

			pair = torch.empty([1, 2*self.channels_KITTI, self.height_KITTI, self.width_KITTI, ])
			
			pair[0] = torch.cat((img1,img2),0)
			
			inputTensor = (pair.float()).cuda()

			# Load the poses. The frames are  0 based indexed.
			poses = open("/data/milatmp1/sharmasa/"+ dataset + "/dataset/poses/" + str(seq).zfill(2) + ".txt").readlines()
			pose_frame1 = np.concatenate( (np.asarray([(float(i)) for i in poses[frame1].split(' ')]).reshape(3,4) , [[0.0,0.0,0.0,1.0]] ), axis=0); # 4x4 transformation matrix
			pose_frame2 = np.concatenate( (np.asarray([(float(i)) for i in poses[frame2].split(' ')]).reshape(3,4) , [[0.0,0.0,0.0,1.0]] ), axis=0); # 4x4 transformation matrix

			# Make the transformation
			pose2_wrt1 = np.dot(inv(pose_frame1),pose_frame2); # Take a point in frame2 to a point in frame 1 ==> point_1 = (pose2_wrt1)*(point_2)

			# Extract R and T, convert it to axis angle form
			R = pose2_wrt1[0:3,0:3]
			axisAngle = (torch.from_numpy(np.asarray(rotMat_to_axisAngle(R))).view(-1,3)).float().cuda()
			

			T = (torch.from_numpy(pose2_wrt1[0:3,3]).view(-1,3)).float().cuda()

			axisAngle = torch.from_numpy(np.array([[0,1,0]])).float().cuda()
			T = torch.from_numpy(np.array([[0.3,0.01,1.5]])).float().cuda()
			return inputTensor,axisAngle,T


