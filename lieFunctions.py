"""
Helper functions for working with the Lie groups SO(3) and SE(3)
"""

import functools
import numpy as np


# # Log map for SO(3). Returns a 3-vector of so(3) exponential coordinates
# # Actually, this does vee(SO3_log(R)), also denoted Log(R) in a few conventions
# # Not to be confused with log(R), which simply returns an element on the tangent space
# def SO3_log(R):


# Convert a rotation matrix to an axis-angle vector. Does vee(SO3_log(rot))
def rotMat_to_axisAngle(rot):

	trace = rot[0,0] + rot[1,1] + rot[2,2]
	trace = np.clip(trace, 0.0, 2.99999)
	theta = np.arccos((trace - 1.0)/2.0)
	omega_cross = (theta/(2*np.sin(theta)))*(rot - np.transpose(rot))

	return [omega_cross[2,1], omega_cross[0,2], omega_cross[1,0]]


# Map an axis-angle vector to a rotation matrix. Does SO3_exp(hat(omega))
def axisAngle_to_rotMat(omega):

	theta = np.sqrt(omega[0]*omega[0] + omega[1]*omega[1] + omega[2]*omega[2])

	if theta < 1e-8:
		return np.eye(3,3)

	omega_cross = np.stack([0.0, -omega[2], omega[1], omega[2], 0.0, -omega[0], -omega[1], omega[0], 0.0])
	omega_cross = np.reshape(omega_cross, [3,3])

	A = np.sin(theta)/theta
	B = (1.0 - np.cos(theta))/(theta**2)
	C = (1.0 - A)/(theta**2)

	omega_cross_square = np.matmul(omega_cross, omega_cross)
	R = np.eye(3,3) + A * omega_cross + B * omega_cross_square
	return R


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


def rotMat_to_euler(M, cy_thresh = None, seq = 'zyx'):
	'''
	Taken From: http://afni.nimh.nih.gov/pub/dist/src/pkundu/meica.libs/nibabel/eulerangles.py
	Discover Euler angle vector from 3x3 matrix
	Uses the conventions above.
	Parameters
	----------
	M : array-like, shape (3,3)
	cy_thresh : None or scalar, optional
	 threshold below which to give up on straightforward arctan for
	 estimating x rotation.  If None (default), estimate from
	 precision of input.
	Returns
	-------
	z : scalar
	y : scalar
	x : scalar
	 Rotations in radians around z, y, x axes, respectively
	Notes
	-----
	If there was no numerical error, the routine could be derived using
	Sympy expression for z then y then x rotation matrix, which is::
	[                       cos(y)*cos(z),                       -cos(y)*sin(z),         sin(y)],
	[cos(x)*sin(z) + cos(z)*sin(x)*sin(y), cos(x)*cos(z) - sin(x)*sin(y)*sin(z), -cos(y)*sin(x)],
	[sin(x)*sin(z) - cos(x)*cos(z)*sin(y), cos(z)*sin(x) + cos(x)*sin(y)*sin(z),  cos(x)*cos(y)]
	with the obvious derivations for z, y, and x
	 z = atan2(-r12, r11)
	 y = asin(r13)
	 x = atan2(-r23, r33)
	for x,y,z order
	y = asin(-r31)
	x = atan2(r32, r33)
	z = atan2(r21, r11)
	Problems arise when cos(y) is close to zero, because both of::
	 z = atan2(cos(y)*sin(z), cos(y)*cos(z))
	 x = atan2(cos(y)*sin(x), cos(x)*cos(y))
	will be close to atan2(0, 0), and highly unstable.
	The ``cy`` fix for numerical instability below is from: *Graphics
	Gems IV*, Paul Heckbert (editor), Academic Press, 1994, ISBN:
	0123361559.  Specifically it comes from EulerAngles.c by Ken
	Shoemake, and deals with the case where cos(y) is close to zero:
	See: http://www.graphicsgems.org/
	The code appears to be licensed (from the website) as "can be used
	without restrictions".
	'''
	M = np.asarray(M)
	if cy_thresh is None:
		try:
			cy_thresh = np.finfo(M.dtype).eps * 4
		except ValueError:
			cy_thresh = _FLOAT_EPS_4
	r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
	# cy: sqrt((cos(y)*cos(z))**2 + (cos(x)*cos(y))**2)
	cy = np.sqrt(r33*r33 + r23*r23)
	if seq=='zyx':
		if cy > cy_thresh: # cos(y) not close to zero, standard form
			z = np.arctan2(-r12,  r11) # atan2(cos(y)*sin(z), cos(y)*cos(z))
			y = np.arctan2(r13,  cy) # atan2(sin(y), cy)
			x = np.arctan2(-r23, r33) # atan2(cos(y)*sin(x), cos(x)*cos(y))
		else: # cos(y) (close to) zero, so x -> 0.0 (see above)
			# so r21 -> sin(z), r22 -> cos(z) and
			z = np.arctan2(r21,  r22)
			y = np.arctan2(r13,  cy) # atan2(sin(y), cy)
			x = 0.0
	elif seq=='xyz':
		if cy > cy_thresh:
			y = np.arctan2(-r31, cy)
			x = np.arctan2(r32, r33)
			z = np.arctan2(r21, r11)
		else:
			z = 0.0
			if r31 < 0:
				y = np.pi/2
				x = arctan2(r12, r13)
			else:
				y = -np.pi/2
	else:
		raise Exception('Sequence not recognized')
	return z, y, x


def euler_to_rotMat(z = 0, y = 0, x = 0, isRadian = True, seq = 'zyx'):
	''' Return matrix for rotations around z, y and x axes
	Uses the z, then y, then x convention above
	Parameters
	----------
	z : scalar
		 Rotation angle in radians around z-axis (performed first)
	y : scalar
		 Rotation angle in radians around y-axis
	x : scalar
		 Rotation angle in radians around x-axis (performed last)
	Returns
	-------
	M : array shape (3,3)
		 Rotation matrix giving same rotation as for given angles
	Examples
	--------
	>>> zrot = 1.3 # radians
	>>> yrot = -0.1
	>>> xrot = 0.2
	>>> M = euler2mat(zrot, yrot, xrot)
	>>> M.shape == (3, 3)
	True
	The output rotation matrix is equal to the composition of the
	individual rotations
	>>> M1 = euler2mat(zrot)
	>>> M2 = euler2mat(0, yrot)
	>>> M3 = euler2mat(0, 0, xrot)
	>>> composed_M = np.dot(M3, np.dot(M2, M1))
	>>> np.allclose(M, composed_M)
	True
	You can specify rotations by named arguments
	>>> np.all(M3 == euler2mat(x=xrot))
	True
	When applying M to a vector, the vector should column vector to the
	right of M.  If the right hand side is a 2D array rather than a
	vector, then each column of the 2D array represents a vector.
	>>> vec = np.array([1, 0, 0]).reshape((3,1))
	>>> v2 = np.dot(M, vec)
	>>> vecs = np.array([[1, 0, 0],[0, 1, 0]]).T # giving 3x2 array
	>>> vecs2 = np.dot(M, vecs)
	Rotations are counter-clockwise.
	>>> zred = np.dot(euler2mat(z=np.pi/2), np.eye(3))
	>>> np.allclose(zred, [[0, -1, 0],[1, 0, 0], [0, 0, 1]])
	True
	>>> yred = np.dot(euler2mat(y=np.pi/2), np.eye(3))
	>>> np.allclose(yred, [[0, 0, 1],[0, 1, 0], [-1, 0, 0]])
	True
	>>> xred = np.dot(euler2mat(x=np.pi/2), np.eye(3))
	>>> np.allclose(xred, [[1, 0, 0],[0, 0, -1], [0, 1, 0]])
	True
	Notes
	-----
	The direction of rotation is given by the right-hand rule (orient
	the thumb of the right hand along the axis around which the rotation
	occurs, with the end of the thumb at the positive end of the axis;
	curl your fingers; the direction your fingers curl is the direction
	of rotation).  Therefore, the rotations are counterclockwise if
	looking along the axis of rotation from positive to negative.
	'''

	if seq != 'xyz' and seq != 'zyx':
		raise Exception('Sequence not recognized')

	if not isRadian:
		z = ((np.pi)/180.) * z
		y = ((np.pi)/180.) * y
		x = ((np.pi)/180.) * x
	if z < -np.pi:
		while z < -np.pi:
			z += 2*np.pi
	if z > np.pi:
		while z > np.pi:
			z -= 2*np.pi
	if y < -np.pi:
		while y < -np.pi:
			y += 2*np.pi
	if y > np.pi:
		while y > np.pi:
			y -= 2*np.pi
	if x < -np.pi:
		while x < -np.pi:
			x += 2*np.pi
	if x > np.pi:
		while x > np.pi:
			x -= 2*np.pi
	assert z>=(-np.pi) and z < np.pi, 'Inapprorpriate z: %f' % z
	assert y>=(-np.pi) and y < np.pi, 'Inapprorpriate y: %f' % y
	assert x>=(-np.pi) and x < np.pi, 'Inapprorpriate x: %f' % x    

	Ms = []

	if seq == 'zyx':

		if z:
				cosz = np.cos(z)
				sinz = np.sin(z)
				Ms.append(np.array(
								[[cosz, -sinz, 0],
								 [sinz, cosz, 0],
								 [0, 0, 1]]))
		if y:
				cosy = np.cos(y)
				siny = np.sin(y)
				Ms.append(np.array(
								[[cosy, 0, siny],
								 [0, 1, 0],
								 [-siny, 0, cosy]]))
		if x:
				cosx = np.cos(x)
				sinx = np.sin(x)
				Ms.append(np.array(
								[[1, 0, 0],
								 [0, cosx, -sinx],
								 [0, sinx, cosx]]))
		if Ms:
				return functools.reduce(np.dot, Ms[::-1])
		return np.eye(3)

	elif seq == 'xyz':

		if x:
				cosx = np.cos(x)
				sinx = np.sin(x)
				Ms.append(np.array(
								[[1, 0, 0],
								 [0, cosx, -sinx],
								 [0, sinx, cosx]]))
		if y:
				cosy = np.cos(y)
				siny = np.sin(y)
				Ms.append(np.array(
								[[cosy, 0, siny],
								 [0, 1, 0],
								 [-siny, 0, cosy]]))
		if z:
				cosz = np.cos(z)
				sinz = np.sin(z)
				Ms.append(np.array(
								[[cosz, -sinz, 0],
								 [sinz, cosz, 0],
								 [0, 0, 1]]))

		if Ms:
				return functools.reduce(np.dot, Ms[::-1])
		return np.eye(3)
