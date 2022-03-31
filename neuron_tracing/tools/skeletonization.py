import cv2 as cv
import tifffile
import numpy as np
import glob
from skimage import morphology

# img_name = '16'
img_name = 'single_branch_noise_1'

org_dir = 'data/single_branch/' + img_name + '.tif'
seg_dir = 'data/single_branch/' + img_name+'.seg.tif'
skel_dir = 'data/single_branch/' + img_name+'.skl.tif'

img_org = tifffile.imread(org_dir)

# 阈值分割
seg = np.zeros([img_org.shape[0],img_org.shape[1],img_org.shape[2]],dtype=np.uint8)
# !!!!!!!!!!!!!!!!!!!!!
seg[img_org>5]=1

# 移除噪声
# seg = morphology.remove_small_objects(seg, min_size=64, connectivity=1)

# 膨胀
kernel_1 = morphology.ball(1)
kernel_2 = morphology.ball(2)
# seg = morphology.dilation(seg, kernel_2)
seg = morphology.erosion(seg, kernel_1)
tifffile.imsave(seg_dir, seg)

#骨架化
skel = morphology.skeletonize_3d(seg)


tifffile.imsave(skel_dir, skel)