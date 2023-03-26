import os
import cv2

# label_path = '/home/group1/zavieton/yolov5/datasets_kitti/kitti/labels/val/000026.txt'
# img_path = '/home/group1/zavieton/yolov5/datasets_kitti/kitti/images/val/000026.png'

label_path = '/home/group1/zavieton/poppy/yolov5-5/data/plant_data/val/labels/gq_low_027.txt'
img_path = '/home/group1/zavieton/poppy/yolov5-5/data/plant_data/val/images/gq_low_027.JPG'

img = cv2.imread(img_path)
[h,w,c] = img.shape
f = open(label_path, 'r')

line = f.readline().split(' ')
for i in range(16):
    try:
        (x1,y1) = (w*(float(line[1])-float(line[3])/2), h*(float(line[2])-float(line[4])/2))
        (x2,y2) = (w*(float(line[1])+float(line[3])/2), h*(float(line[2])+float(line[4])/2))
        print(x1,y1,x2,y2)

        cv2.rectangle(img, (int(x1),int(y1)), (int(x2),int(y2)), (255,0,0), 20)
        line = f.readline().split(' ')
    except:
        break

img = cv2.resize(img, (640,640))
cv2.imwrite('1.png', img)

# [415 564  20  30 261]
# [519 533  19  46 282]
# [473 441  30  33 286]
# [521 415  22  34 302]
# [228 483  22  50 368]
# [229 535  19  35 382]
# [484 568  21  41 437]
# [443 511  27  29 462]
# [503 562  31  57 576]
# [474 508  54  53 599]
# [495 467  40  45 686]
# [502 369  43  57 809]
# [335 546  36  51 831]
# [455 467  39  53 851]
# [ 368  507   34   77 1226]
# [ 156  478   70   95 3110]