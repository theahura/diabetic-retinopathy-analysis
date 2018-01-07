import glob

from PIL import Image, ImageChops, ImageOps


fp = 'train_2/*.jpeg'
for i, f in enumerate(glob.glob(fp)):
    crop = Image.open(f)
    crop = crop.resize((512, 512), Image.ANTIALIAS)
    crop.save(f)
    print f
