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

fp = 'test/*.jpeg'

size = 512, 512
scale = 500

def scaleRadius(img, scale):
    x = img[img.shape[0]/2, :, :].sum(1)
    r = (x > x.mean()/10).sum()/2
    s = scale*1.0/r
    return cv2.resize(img, (0, 0), fx=s, fy=s)


for i, f in enumerate(glob.glob(fp)):
    try:
        a = cv2.imread(f)

        # Radius crop and color balance.
        a = scaleRadius(a, scale)
        b = np.zeros(a.shape)
        cv2.circle(b, (a.shape[1]/2, a.shape[0]/2), int(scale*0.9), (1, 1, 1), -1, 8, 0)
        aa = cv2.addWeighted(a, 4, cv2.GaussianBlur(a, (0, 0), scale/30), -4, 128)*b + 128*(1 - b)

        # Remove border.
        image = Image.fromarray(np.uint8(aa))
        diff = max(image.size) - min(image.size)
        crop_size = diff/2

        if image.size[0] > image.size[1]:
            crop = image.crop(
                (
                    crop_size,
                    0,
                    image.size[0] - crop_size,
                    image.size[1]
                )
            )
        else:
            crop = image.crop(
                (
                    0,
                    crop_size,
                    image.size[0],
                    image.size[1] - crop_size
                )
            )

        # Handle rounding issues.
        max_side = max(crop.size)
        crop = crop.resize((max_side, max_side), Image.ANTIALIAS)

        # Resize.
        crop.thumbnail(size, Image.ANTIALIAS)

        # Save.
        crop.save(f)
    except:
        print i
        print f
