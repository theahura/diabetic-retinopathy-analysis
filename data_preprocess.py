"""
Scales, cuts radius, and color balances all data in ./data folder.

See code at
https://github.com/btgraham/SparseConvNet/blob/kaggle_Diabetic_Retinopathy_competition/Data/kaggleDiabeticRetinopathy/preprocessImages.py

See code at: https://github.com/JeffreyDF/kaggle_diabetic_retinopathy/blob/master/generators.py
"""

import glob

import cv2
import IPython
import numpy as np
from PIL import Image, ImageChops, ImageOps

DO_TEST = False
scale = 180

if DO_TEST:
    fp = 'test/*.jpeg'
else:
    fp = 'train/*.jpeg'

size = 512, 512
for i, f in enumerate(glob.glob(fp)):
    # Resize the image.
    image = Image.open(f)
    image.thumbnail(size, Image.ANTIALIAS)
    image_size = image.size
    thumb = image.crop((0, 0, size[0], size[1]))
    offset_x = max((size[0] - image_size[0]) / 2, 0)
    offset_y = max((size[1] - image_size[1]) / 2, 0)
    thumb = ImageChops.offset(thumb, offset_x, offset_y)

    # Radius crop and color balance the image.
    a = np.asarray(thumb)
    b = np.zeros(a.shape)
    cv2.circle(b, (a.shape[1]/2, a.shape[0]/2), int(scale*0.9), (1, 1, 1), -1, 8, 0)
    aa = cv2.addWeighted(a, 4, cv2.GaussianBlur(a, (0, 0), scale/30), -4, 128)*b+128*(1-b)

    # Save.
    cv2.imwrite(f, aa)

    print i
    print f
