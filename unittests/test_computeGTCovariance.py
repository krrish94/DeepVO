# Unit test for helpers.py

import sys
import os
sys.path.append(str(os.path.join(os.path.dirname(sys.argv[0]), '..')))

from KITTIDataset import KITTIDataset
import numpy as np
import time
from torch.utils.data import DataLoader
from tqdm import trange, tqdm


if __name__ == '__main__':

	train_seq = [0, 1, 2, 8, 9]
	train_startFrames = [0, 0, 0, 0, 0]
	train_endFrames = [4540, 1100, 4660, 4070, 1590]
	# train_seq = [1]
	# train_startFrames = [0]
	# train_endFrames = [40]

	kitti_dataset = KITTIDataset('/data/milatmp1/jatavalk/KITTIOdometry/dataset/', sequences = train_seq, \
		startFrames = train_startFrames, endFrames = train_endFrames, parameterization = 'mahalanobis')
	
	start_time = time.time()
	# dataLoader = DataLoader(kitti_dataset, batch_size = 1, shuffle = False, num_workers = 1)
	gt_all = np.zeros((len(kitti_dataset), 6), dtype = np.float32)
	i = 0
	for sample in tqdm(kitti_dataset):
		_, gt, _, _, _, _, _ = sample
		gt_all[i] = gt.cpu().numpy()
		i += 1
	print('--- %s seconds ---' % (time.time() - start_time))
	print(np.cov(gt_all.T))
	infoMat = np.linalg.inv(np.cov(gt_all.T))
	np.savetxt('infoMat.txt', infoMat)
