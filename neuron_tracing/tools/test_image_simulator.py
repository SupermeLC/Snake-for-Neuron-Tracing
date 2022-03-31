import numpy as np
from klib.baseio import *
from klib.simulator import *
from klib.glib.Draw3DTools import *
from skimage import feature
from pylab import *
import copy
import random
import cv2 as cv
import multiprocessing as mp
import tifffile

#size,max_depth,base_length,base_x,mean,std
#384，[4,5,6], [40,50,60], 1, [25,35],[8,12]
#simulator(size,max_depth,base_length,base_x,mean,std):



def main(j):
    np.random.seed()
    print(j)
    k=1
    data_type_ = np.uint8


    external = False
    internal = True

    if external == True:
        add_point_noise = True
        blur_ = True
        if data_type_ == np.uint16:
            mean = np.random.randint(0,200)
            std = np.random.randint(15,25)
        else:
            mean = np.random.randint(0,20)
            std = np.random.randint(0,10)
    else:
        add_point_noise = False
        blur_ = True
        mean = 0
        std = 0

    max_depth = 3
    base_length = 20

    # max_depth = np.random.randint(0,1)
    # base_length = np.random.randint(0,1)
    # mean = np.random.randint(130,170)
    # std = np.random.randint(15,25)
    # max_depth = np.random.randint(4,7)
    # base_length = np.random.randint(40,60)
    # mean = np.random.randint(0,20)
    # std = np.random.randint(0,10)

    print(max_depth,base_length,mean,std)
    if external == True:
        raw, img, _ = simulator(255, max_depth, base_length, 1, mean, std, addnoise=add_point_noise,blur = blur_, data_type=data_type_)
        label = np.zeros([img.shape[0], img.shape[1], img.shape[2]])
        label[raw > 10] = 1
    else:
        img, _, _ = simulator(64, max_depth, base_length, 1, mean, std, addnoise=add_point_noise,blur = blur_, data_type=data_type_)
        # label = np.zeros([img.shape[0], img.shape[1], img.shape[2]])
        # label[img > 10] = 1

    image_dir = 'data/single_branch/single_branch_noise_1.tif'
    tifffile.imsave(image_dir, img)
    




if __name__ == '__main__':
    
    pool = mp.Pool()  # 定义一个Pool，并定义CPU核数量为10
    pool.map(main, range(0, 1))