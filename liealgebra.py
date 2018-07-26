import numpy as np

def rotMat_to_axisAngle(rot):
		# qt = Quaternion(matrix=rot)
		# axis = qt.axis
		# angle = qt.radians
		#return axis,angle

		trace = rot[0,0] + rot[1,1] + rot[2,2]
		trace = np.clip(trace, 0.0, 2.99999)
		theta = np.arccos((trace - 1.0)/2.0)
		omega_cross = (theta/(2*np.sin(theta)))*(rot - np.transpose(rot))
		
		return [omega_cross[2,1], omega_cross[0,2], omega_cross[1,0]]


# Expects axis in column vector format
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
	R = np.eye(3,3) + A*omega_cross + B*omega_cross_square
	return R



