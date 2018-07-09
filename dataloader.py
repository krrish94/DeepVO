from skimage import io, transform
from skimage.transform import resize
from numpy.linalg import inv
from numpy import matmul
from pyquaternion import Quaternion

import numpy as np
import os
import random as rn
import scipy.misc as smc
import pandas as pd
import torch



class Dataloader:
	def __init__(self):

		
		# KITTI dataloader parameters
		self.r_KITTImean = 88.61
		self.g_KITTImean = 93.70
		self.b_KITTImean = 92.11
		
		self.train_seqs_KITTI = [0,1,2,8,9]
		self.test_seqs_KITTI = [3,4,5,6,7,10]

		self.minFrame_KITTI = 5;
		self.maxFrame_KITTI = 500;

		# Dimensions to be fed in the input
		self.width_KITTI = 1280;
		self.height_KITTI = 384;
		self.channels_KITTI = 3;

	# Get start and end of a subsequence 
	def getSubsequence(self,seq,tl,dataset):
		if dataset == "KITTI":
			seqLength = len(os.listdir("/data/milatmp1/sharmasa/"+ dataset + "/dataset/sequences/" + str(seq).zfill(2) + "/image_2/"))

		st_frm = rn.randint(1, seqLength-tl+1)
		end_frm = st_frm + tl - 1;
		return st_frm, end_frm

	def rotMat_to_axisAngle(self,rot):
		# qt = Quaternion(matrix=rot)
		# axis = qt.axis
		# angle = qt.radians
		#return axis,angle

		trace = rot[0,0] + rot[1,1] + rot[2,2]
		trace = np.clip(trace, 0.0, 2.99999)
		theta = np.arccos((trace - 1.0)/2.0)
		omega_cross = (theta/(2*np.sin(theta)))*(rot - np.transpose(rot))
		return [omega_cross[2,1], omega_cross[0,2], omega_cross[1,0]]


	def preprocessImg(self,img):
		# Subtract the mean R,G,B pixels
		img[:,:,0] = img[:,:,0] - self.r_KITTImean 
		img[:,:,1] = img[:,:,1] - self.g_KITTImean 
		img[:,:,2] = img[:,:,2] - self.b_KITTImean
	
		# Resize to the dimensions required 
		img = np.resize(img,(self.height_KITTI,self.width_KITTI,self.channels_KITTI))
		

		# Change the channel
		newImg = np.empty([self.channels_KITTI,self.height_KITTI,self.width_KITTI])
		newImg[0,:,:] = img[:,:,0]
		newImg[1,:,:] = img[:,:,1]
		newImg[2,:,:] = img[:,:,2]
		
		return newImg


		


	# Get the image pair and their corresponding R and T.
	def getPairFrameInfo(self,frm1,frm2,seq,dataset):

		if dataset == "KITTI":
			# Load the two images : loaded as H x W x 3(R,G,B)
			img1 = smc.imread("/data/milatmp1/sharmasa/"+ dataset + "/dataset/sequences/" + str(seq).zfill(2) + "/image_2/" + str(frm1).zfill(6) + ".png")
			img2 = smc.imread("/data/milatmp1/sharmasa/"+ dataset + "/dataset/sequences/" + str(seq).zfill(2) + "/image_2/" + str(frm2).zfill(6) + ".png")
			
			# Preprocess
			img1 = self.preprocessImg(img1)
			img2 = self.preprocessImg(img2)

			pair = np.empty([1, 2*self.channels_KITTI, self.height_KITTI, self.width_KITTI, ])
			
			pair[0] = np.append(img1,img2,axis=0)

			inputTensor = (torch.from_numpy(pair).float()).cuda()

			# Load the poses. The frames are  0 based indexed.
			poses = open("/data/milatmp1/sharmasa/"+ dataset + "/dataset/poses/" + str(seq).zfill(2) + ".txt").readlines()
			pose_frm1 = np.append(np.asarray([(float(i)) for i in poses[frm1].split(' ')]).reshape(3,4),[[0,0,0,1]], axis=0); # 4x4 transformation matrix
			pose_frm2 = np.append(np.asarray([(float(i)) for i in poses[frm2].split(' ')]).reshape(3,4),[[0,0,0,1]], axis=0); # 4x4 transformation matrix

			# Make the transformation
			pose2_wrt1 = matmul(inv(pose_frm1),pose_frm2);
			# Extract R and T, convert it to axis angle form
			R = pose2_wrt1[0:3,0:3]
			axis = (torch.from_numpy(np.asarray(self.rotMat_to_axisAngle(R))).view(-1,3)).float().cuda()

			T = (torch.from_numpy(pose2_wrt1[0:3,3]).view(-1,3)).float().cuda()
			
			
			

			return inputTensor,axis,T


