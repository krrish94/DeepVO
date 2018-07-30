# Unit test for helpers.py

import sys
import os
print(sys.path)
sys.path.append(str(os.path.join(os.path.dirname(sys.argv[0]), '..')))
print(sys.path)

from helpers import firstGE


if __name__ == '__main__':

	# Test firstGE
	egList = [1, 1, 1, 2, 2, 2, 2, 2, 2]
	print(firstGE(egList, 2))
	egList2 = [2, 3, 5, 8]
	for i in range(8):
		print(firstGE(egList2, i))
