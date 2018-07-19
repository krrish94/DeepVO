import sys
import numpy as np
import numpy as np
import os
import random as rn
import scipy.misc as smc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from liealgebra import rotMat_to_axisAngle


def checkRotMattoAxisAngle():
	# Check for axis angle to rotation matrix
	axisAngle_file = open("./matlab/axisData.txt",'r')
	gt_file = open('./matlab/01.txt')

	num_lines_ax = sum(1 for line in open('./matlab/axisData.txt'))
	num_lines_gt = sum(1 for line in open('./matlab/01.txt'))
	assert num_lines_ax == num_lines_gt

	axisAngles = np.empty([num_lines_gt,3])
	i=0
	for line in axisAngle_file:
		ax_= line.split()
		
		ax_gt = np.empty([1,3])
		ax_gt[0,0] = float(ax_[0]);
		ax_gt[0,1] = float(ax_[1]) 
		ax_gt[0,2] = float(ax_[2])
		axisAngles[i,:] = ax_gt
		i=i+1
		
		
	axisAngle_file.close()
	i=0
	gt_file = open('./matlab/01.txt')
	cross_prod=[]
	for line in gt_file:
		t = line.split() # r11 r12 r13 t1 r21 r22 r23 t2 r31 r32 r33 t3
		rot = np.empty([3,3])
		
		rot[0,0] = float(t[0]) 
		rot[0,1] = float(t[1]) 
		rot[0,2] = float(t[2]) 

		rot[1,0] = float(t[4]) 
		rot[1,1] = float(t[5]) 
		rot[1,2] = float(t[6])

		rot[1,0] = float(t[8]) 
		rot[1,1] = float(t[9]) 
		rot[1,2] = float(t[10])


		ax_=rotMat_to_axisAngle(rot)

		ax_est = np.empty([1,3])
		ax_est[0,0] = ax_[0]
		ax_est[0,1] = ax_[1]
		ax_est[0,2] = ax_[2]

		cross_prod.append(np.linalg.norm(np.cross(axisAngles[i,:], ax_est)))

	gt_file.close()

	fig,ax = plt.subplots(1)
	ax.plot(cross_prod,'r', label="cross product norm")
	plt.ylabel(" Cross product norm ")
	plt.xlabel(" num samples ")
	plt.ylim(-0.001,0.001)
	fig.savefig("/u/sharmasa/Documents/DeepVO/matlab/axis-angleTest")


def computeMeanandStddevValue():
	print("Computing mean ==> ")
	mean=[]
	for seq in range(11):
		print(seq)
		for frm  in range(len(os.listdir("/data/milatmp1/sharmasa/"+ "KITTI" + "/dataset/sequences/" + str(seq).zfill(2) + "/image_2/"))):
			img = smc.imread("/data/milatmp1/sharmasa/"+ "KITTI" + "/dataset/sequences/" + str(seq).zfill(2) + "/image_2/" + str(frm).zfill(6) + ".png")
			mean.append(np.mean(img,axis=(0,1)))

	mean = np.mean(mean,axis=0)
	print("mean is (R, G, B) : ",  mean)
	print("Computing std dev ==> ")
	stddev=[];
	for seq in range(11):
		print(seq)
		for frm  in range(len(os.listdir("/data/milatmp1/sharmasa/"+ "KITTI" + "/dataset/sequences/" + str(seq).zfill(2) + "/image_2/"))):
			img = smc.imread("/data/milatmp1/sharmasa/"+ "KITTI" + "/dataset/sequences/" + str(seq).zfill(2) + "/image_2/" + str(frm).zfill(6) + ".png")
			r = img[:,:,0] - mean[0]
			g = img[:,:,1] - mean[1]
			b = img[:,:,2] - mean[2]
			
			r = np.square(r)
			g = np.square(g)
			b = np.square(b)
			stddev.append([np.mean(r),np.mean(g),np.mean(b)])

	stddev = np.sqrt(np.mean(stddev,axis=0));
	print("Stddev is (R,G,B) : " , stddev)


#checkRotMattoAxisAngle()
computeMeanandStddevValue()
