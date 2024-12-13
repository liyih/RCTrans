import cv2
import os.path as osp
import os
import pdb
import numpy as np
from tqdm import tqdm
pic_list = os.listdir('result_vis/swint')
for index in tqdm(range(len(pic_list))):
    pic = cv2.imread('result_vis/swint/'+pic_list[index])
    
    radar = pic[0:630, 0:800]
    radar = cv2.resize(radar, (1600, 1200))
    b_l = pic[1785:2385, 20:800]
    b_l = cv2.resize(b_l, (800, 600))
    f_l = pic[1775:2375, 800:1580]
    f_l = cv2.resize(f_l, (800, 600))
    f = pic[600:1200, 20:800]
    f = cv2.resize(f, (800, 600))
    f_r = pic[600:1200, 800:1580]
    f_r = cv2.resize(f_r, (800, 600))
    b_r = pic[1200:1800, 20:800]
    b_r = cv2.resize(b_r, (800, 600))
    b = pic[1200:1800, 800:1580]
    b = cv2.resize(b, (800, 600))
    canvas = np.zeros([1200, 4000, 3])
    canvas[0:1200,0:1600] = radar
    canvas[0:600,1600:2400] = f_l
    canvas[0:600,2400:3200] = f
    canvas[0:600,3200:4000] = f_r
    canvas[600:1200,1600:2400] = b_l
    canvas[600:1200,2400:3200] = b
    canvas[600:1200,3200:4000] = b_r
    cv2.imwrite('new_pic/'+str(index)+'.jpg', canvas[20:,:,:])