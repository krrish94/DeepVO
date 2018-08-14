from lieFunctions import axisAngle_to_rotMat, quat_to_rotMat, euler_to_rotMat
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import glob


import warnings
warnings.filterwarnings("ignore")
	

def getGroundTruthTrajectory(seq, seqLength, dataDir):
	
	cameraTraj = np.empty([seqLength,3])
	# poses = open(os.path.join(dataDir, 'poses', str(seq).zfill(2) + '.txt'))
	poses = np.loadtxt(os.path.join(dataDir, 'poses', str(seq).zfill(2) + '.txt'))
	# poses = open("/data/milatmp1/sharmasa/"+ dataset + "/dataset/poses/" + str(seq).zfill(2) + ".txt").readlines()
	for frame in range(seqLength):
		# pose = np.concatenate((np.asarray([(float(i)) for i in poses[frame].split(' ')]).reshape(3,4) , [[0.0,0.0,0.0,1.0]]), axis=0);
		pose = np.concatenate((np.asarray(poses[frame]).reshape(3, 4), [[0., 0., 0., 1.]]), axis = 0)
		cameraTraj[frame,:] = pose[0:3,3].T;

	return cameraTraj;


def plotSequenceRelative(expDir, seq, seqLength, trajectory, dataDir, cmd, epoch):

	T = np.eye(4);
	estimatedCameraTraj = np.empty([seqLength,3])

	# Extract the camera centres from all the frames

	# First frame as the world origin
	estimatedCameraTraj[0] = np.zeros([1,3]);
	for frame in range(seqLength-1):

		# Output is pose of frame i+1 with respect to frame i
		relativePose = trajectory[frame,:]
		
		if cmd.outputParameterization == 'default':
			R = axisAngle_to_rotMat(np.transpose(relativePose[:3]))
			t = np.reshape(relativePose[3:],(3,1))
		elif cmd.outputParameterization == 'quaternion':
			R = quat_to_rotMat(relativePose[:4])
			t = np.reshape(relativePose[4:],(3,1))
		elif cmd.outputParameterization == 'euler':
			R = np.asarray(euler_to_rotMat(x = relativePose[0], y = relativePose[1], \
				z = relativePose[3], seq = 'xyz'), dtype = np.float32)
			t = (np.asarray(relativePose[3:], dtype = np.float32)).reshape((3,1))

		T_r = np.concatenate( ( np.concatenate([R,t],axis=1) , [[0.0,0.0,0.0,1.0]] ) , axis = 0 )
		
		# With respect to the first frame
		T_abs = np.dot(T,T_r);
		# Update the T matrix till now.
		T = T_abs

		# Get the origin of the frame (i+1), ie the camera center
		estimatedCameraTraj[frame+1] = np.transpose(T[0:3,3])

	# Get the ground truth camera trajectory
	gtCameraTraj = getGroundTruthTrajectory(seq, seqLength, dataDir);

	# Plot the estimated and groundtruth trajectories
	x_gt = gtCameraTraj[:,0]
	z_gt = gtCameraTraj[:,2]

	x_est = estimatedCameraTraj[:,0]
	z_est = estimatedCameraTraj[:,2]
	
	# Save plot
	# path = os.path.join(expDir, 'plots', 'traj', str(seq).zfill(2))
	# currNumPlots = len(glob.glob1(path,"*.png"))

	# assert (currNumPlots%2==0)
	fig,ax = plt.subplots(1)
	ax.plot(x_gt,z_gt, 'c', label = "ground truth")
	ax.plot(x_est,z_est, 'm', label= "estimated")
	ax.legend()
	# fig.savefig(path + "/" + str(currNumPlots/2 + 1))
	fig.savefig(os.path.join(expDir, 'plots', 'traj', str(seq).zfill(2), 'traj_' + str(epoch).zfill(3)))

	# # To save only the predicted plot
	# fig_,ax_ = plt.subplots(1)
	# ax_.plot(x_est,z_est, 'm', label="estimated traj")
	# ax_.legend()
	# fig_.savefig(path + "/est_" + str(currNumPlots/2 + 1))


def plotSequenceAbsolute(expDir, seq, seqLength, trajectory, dataDir, cmd, epoch):

	T = np.eye(4);
	estimatedCameraTraj = np.empty([seqLength,3])

	# Extract the camera centres from all the frames

	for frame in range(seqLength-1):

		# Output is pose of frame i+1 with respect to frame i
		absolutePose = trajectory[frame,:]
		
		if cmd.outputParameterization == 'default':
			R = axisAngle_to_rotMat(np.transpose(absolutePose[:3]))
			t = np.reshape(absolutePose[3:],(3,1))
		elif cmd.outputParameterization == 'quaternion':
			R = quat_to_rotMat(absolutePose[:4])
			t = np.reshape(absolutePose[4:],(3,1))
		elif cmd.outputParameterization == 'euler':
			R = np.asarray(euler_to_rotMat(x = absolutePose[0], y = absolutePose[1], \
				z = absolutePose[3], seq = 'xyz'), dtype = np.float32)
			t = (np.asarray(absolutePose[3:], dtype = np.float32)).reshape((3,1))

		T_r = np.concatenate( ( np.concatenate([R,t],axis=1) , [[0.0,0.0,0.0,1.0]] ) , axis = 0 )
		
		# Get the origin of the frame (i+1), ie the camera center
		estimatedCameraTraj[frame+1] = np.transpose(T_r[0:3,3])

	# Get the ground truth camera trajectory
	gtCameraTraj = getGroundTruthTrajectory(seq, seqLength, dataDir);

	# Plot the estimated and groundtruth trajectories
	x_gt = gtCameraTraj[:,0]
	z_gt = gtCameraTraj[:,2]

	x_est = estimatedCameraTraj[:,0]
	z_est = estimatedCameraTraj[:,2]

	# assert (currNumPlots%2==0)
	fig,ax = plt.subplots(1)
	ax.plot(x_gt,z_gt, 'c', label = "ground truth")
	ax.plot(x_est,z_est, 'm', label= "estimated")
	ax.legend()
	fig.savefig(os.path.join(expDir, 'plots', 'traj', str(seq).zfill(2), 'traj_' + str(epoch).zfill(3)))









	



