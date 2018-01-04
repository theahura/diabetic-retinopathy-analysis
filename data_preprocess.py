"""
Scales, cuts radius, and color balances all data in ./data folder.

See code at
https://github.com/btgraham/SparseConvNet/blob/kaggle_Diabetic_Retinopathy_competition/Data/kaggleDiabeticRetinopathy/preprocessImages.py

See code at: https://github.com/JeffreyDF/kaggle_diabetic_retinopathy/blob/master/generators.py
"""

import cv2
import glob
import numpy
import IPython


def scaleRadius(img, scale):
    x = img[img.shape[0]/2, :, :].sum(1)
    r = (x > x.mean()/10).sum()/2
    s = scale*1.0/r
    return cv2.resize(img, (0, 0), fx=s, fy=s)

scale = 300
for f in (glob.glob("train_2/*.jpeg")):
    a = cv2.imread(f)
    a = scaleRadius(a,scale)
    b = numpy.zeros(a.shape)
    cv2.circle(b, (a.shape[1]/2, a.shape[0]/2), int(scale*0.9), (1, 1, 1), -1, 8, 0)
    aa = cv2.addWeighted( a, 4, cv2.GaussianBlur(a, (0, 0), scale/30), -4, 128)*b+128*(1 - b)
    cv2.imwrite(f)
