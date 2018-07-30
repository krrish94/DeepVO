# Unit test for helpers.py

import sys
import os
sys.path.append(str(os.path.join(os.path.dirname(sys.argv[0]), '..')))

from KITTIDataset import KITTIDataset
import time
from torch.utils.data import DataLoader


if __name__ == '__main__':

	# kitti_dataset = KITTIDataset('kittipath', sequences = [1,3,2,4], startFrames = [0, 0, 0, 0], \
	# endFrames = [10000, 10000, 10000, 10000])
	kitti_dataset = KITTIDataset('/data/milatmp1/sharmasa/KITTI/dataset/', sequences = [1,3,2,4], \
		startFrames = [2,4,100,52], endFrames = [6,6,101,56], parameterization = 'euler	')
	# for i in range(len(kitti_dataset)):
	# 	print(i)
	# 	print(kitti_dataset[i])

	start_time = time.time()
	dataLoader = DataLoader(kitti_dataset, batch_size = 1, shuffle = False, num_workers = 1)
	for i in dataLoader:
		# print(i)
		# print(i[0].item(), i[1].item(), i[2].item())
		pass
	print('--- %s seconds ---' % (time.time() - start_time))
