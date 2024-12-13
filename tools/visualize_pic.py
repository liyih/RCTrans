import cv2
import numpy as np
import pdb
gt = cv2.imread('result_vis/gt/1dfecb8189f54b999f4e47ddaa677fd0_gt.png')
res18 = cv2.imread('result_vis/res18/1dfecb8189f54b999f4e47ddaa677fd0_pred.png')
res50 = cv2.imread('result_vis/res50/1dfecb8189f54b999f4e47ddaa677fd0_pred.png')
swint = cv2.imread('result_vis/swint/1dfecb8189f54b999f4e47ddaa677fd0_pred.png')
# pdb.set_trace()

# gt_radar = gt[50:490, 110:705]
# res18_radar = res18[50:490, 110:705]
# res50_radar = res50[50:490, 110:705]
# swint_radar = swint[50:490, 110:705]

# canvas = np.zeros([440, 2380, 3])
# canvas[:, 0:595] = gt_radar
# canvas[:, 595:1190] = res18_radar
# canvas[:, 1190:1785] = res50_radar
# canvas[:, 1785:2380] = swint_radar
# cv2.imwrite('a.jpg', canvas)

# gt_1 = gt[1870:2315, 20:800]
# res18_1 = res18[1870:2315, 20:800]
# res50_1 = res50[1870:2315, 20:800]
# swint_1 = swint[1870:2315, 20:800]

# canvas = np.zeros([445, 3120, 3])
# canvas[:, 0:780] = gt_1
# canvas[:, 780:1560] = res18_1
# canvas[:, 1560:2340] = res50_1
# canvas[:, 2340:3120] = swint_1

# cv2.imwrite('b.jpg', canvas)

gt_1 = gt[1870:2315, 800:1580]
res18_1 = res18[1870:2315, 800:1580]
res50_1 = res50[1870:2315, 800:1580]
swint_1 = swint[1870:2315, 800:1580]

canvas = np.zeros([445, 3120, 3])
canvas[:, 0:780] = gt_1
canvas[:, 780:1560] = res18_1
canvas[:, 1560:2340] = res50_1
canvas[:, 2340:3120] = swint_1

cv2.imwrite('c.jpg', canvas)