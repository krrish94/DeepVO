import sys
import numpy as np
import pandas as pd
from skimage import io, transform
from skimage.transform import resize
import numpy as np
import os
import random as rn
import scipy.misc as smc

x=[]

for seq in range(11):
	for frm  in range(len(os.listdir("/data/milatmp1/sharmasa/"+ "KITTI" + "/dataset/sequences/" + str(seq).zfill(2) + "/image_2/"))):
		img = smc.imread("/data/milatmp1/sharmasa/"+ "KITTI" + "/dataset/sequences/" + str(seq).zfill(2) + "/image_2/" + str(frm).zfill(6) + ".png")
		x.append(np.mean(img,axis=(0,1)))


print(np.mean(x,axis=0))
