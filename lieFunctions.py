"""
Helper functions for working with the Lie groups SO(3) and SE(3)
"""

import numpy as np


# # Log map for SO(3). Returns a 3-vector of so(3) exponential coordinates
# # Actually, this does vee(SO3_log(R)), also denoted Log(R) in a few conventions
# # Not to be confused with log(R), which simply returns an element on the tangent space
# def SO3_log(R):


# Convert a 3 x 3 rotation matrix to a quaternion
# Abridged from the elegant pyquaternion library 
# https://github.com/KieranWynn/pyquaternion/
def rotMat_to_quat(R):

	# if not np.allclose(np.dot(R.conj().transpose(), R), np.eye(3, dtype = np.double)):
	# 	raise ValueError('Matrix must be orthogonal!')
	# if not np.allclose(np.linalg.det(R), 1.0):
	# 	raise ValueError('Matrix must be special orthogonal! (Det(R) = +1)')

	m = R.conj().transpose()
	if m[2, 2] < 0:
		if m[0, 0] > m[1, 1]:
			t = 1 + m[0, 0] - m[1, 1] - m[2, 2]
			q = [m[1, 2]-m[2, 1],  t,  m[0, 1]+m[1, 0],  m[2, 0]+m[0, 2]]
		else:
			t = 1 - m[0, 0] + m[1, 1] - m[2, 2]
			q = [m[2, 0]-m[0, 2],  m[0, 1]+m[1, 0],  t,  m[1, 2]+m[2, 1]]
	else:
		if m[0, 0] < -m[1, 1]:
			t = 1 - m[0, 0] - m[1, 1] + m[2, 2]
			q = [m[0, 1]-m[1, 0],  m[2, 0]+m[0, 2],  m[1, 2]+m[2, 1],  t]
		else:
			t = 1 + m[0, 0] + m[1, 1] + m[2, 2]
			q = [t,  m[1, 2]-m[2, 1],  m[2, 0]-m[0, 2],  m[0, 1]-m[1, 0]]

	q = np.array(q)
	q *= 0.5 / np.sqrt(t);
	
	return q


def quat_to_rotMat(q):
	''' Calculate rotation matrix corresponding to quaternion
	https://afni.nimh.nih.gov/pub/dist/src/pkundu/meica.libs/nibabel/quaternions.py
	Parameters
	----------
	q : 4 element array-like
	Returns
	-------
	M : (3,3) array
	  Rotation matrix corresponding to input quaternion *q*
	Notes
	-----
	Rotation matrix applies to column vectors, and is applied to the
	left of coordinate vectors.  The algorithm here allows non-unit
	quaternions.
	References
	----------
	Algorithm from
	http://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
	Examples
	--------
	>>> import numpy as np
	>>> M = quat2mat([1, 0, 0, 0]) # Identity quaternion
	>>> np.allclose(M, np.eye(3))
	True
	>>> M = quat2mat([0, 1, 0, 0]) # 180 degree rotn around axis 0
	>>> np.allclose(M, np.diag([1, -1, -1]))
	True
	'''
	w, x, y, z = q
	Nq = w*w + x*x + y*y + z*z
	if Nq < 1e-8:
		return np.eye(3)
	s = 2.0/Nq
	X = x*s
	Y = y*s
	Z = z*s
	wX = w*X; wY = w*Y; wZ = w*Z
	xX = x*X; xY = x*Y; xZ = x*Z
	yY = y*Y; yZ = y*Z; zZ = z*Z
	return np.array([[ 1.0-(yY+zZ), xY-wZ, xZ+wY ], \
		[ xY+wZ, 1.0-(xX+zZ), yZ-wX ], [ xZ-wY, yZ+wX, 1.0-(xX+yY) ]])
