"""
Rescales, cuts radius, and color balances all data in ./data folder.

See code at
https://github.com/btgraham/SparseConvNet/blob/kaggle_Diabetic_Retinopathy_competition/Data/kaggleDiabeticRetinopathy/preprocessImages.py
"""

import cv2
import glob
import numpy

DO_TEST = False

def scaleRadius(img, scale):
    x = img[img.shape[0]/2, :, :].sum(1)
    r = (x > x.mean()/10).sum()/2
    s = scale*1.0/r
    return cv2.resize(img, (0, 0), fx=s, fy=s)

scale = 300

if DO_TEST:
    fp = 'test/*.jpeg'
else:
    fp = 'train/*.jpeg'

for i, f in enumerate(glob.glob(fp)):
    try:
        # Load the image.
        a = cv2.imread(f)
        a = scaleRadius(a, scale)

        # Create a mask.
        b = numpy.zeros(a.shape)
        cv2.circle(b, (a.shape[1]/2, a.shape[0]/2), int(scale*0.9), (1, 1, 1), -1, 8, 0)

        # Apply the mask and the local average subtraction.
        aa = cv2.addWeighted(a, 4, cv2.GaussianBlur(a, (0, 0), scale/30), -4, 128)*b+128*(1-b)

        # Save.
        cv2.imwrite(f, aa)

        print i
        print f
    except:
        print f
