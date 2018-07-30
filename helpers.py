# Script containing helper functions used in a few other files
import math

# Helper method to return the first index in a sorted list that is greater than or equal to
# the given key
def firstGE(sortedList, inputKey):

	# Basically, do a binary search
	low = 0
	high = len(sortedList)
	while low != high:
		mid = int(math.floor((low + high) / 2))
		if sortedList[mid] <= inputKey:
			low = mid + 1
		else:
			high = mid
	# If an element greater than equal to the key exists
	# both low and mid contain the index of that element
	if low == high:
		return low
	else:
		return -1
