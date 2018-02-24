import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import io
import time
# import picamera
import requests
import PIL
from PIL import Image
from easygui import ccbox
#import RPi.GPIO as GPIO
import tempfile
import IPython
import os


    
#Define api urls
SERV_ADDR = "http://drcam.molabs.tech"
UPLOAD_URL = SERV_ADDR + "/upload"
REQUEST_IMAGE_URL = SERV_ADDR + "/image"

try:
    #Load picamera and begin preview
    camera = picamera.PiCamera()
    camera.resolution = (512, 512)
    camera.start_preview()
    stream = io.BytesIO()
    
 #    #Wait for shutter press

	# #Pause before capturing
    time.sleep(2)
    camera.capture(stream, 'jpeg')
    camera.stop_preview()

    # image = Image.open(os.path.expanduser("~/Downloads/217_left.jpeg"))

    #Process image and save in temp file
    image = Image.open(stream)
    orig_size = image.size
    image.thumbnail((512, 512), PIL.Image.ANTIALIAS)
    tf = tempfile.NamedTemporaryFile()
    image.save(tf, format='jpeg')

    #Display selection GUI
    print "Loading GUI"
    reply = ccbox("Do you want to analyze this image?", "Image Viewer", 
        image=tf.name)

    #Resize image back to original size for server
    image = image.resize(orig_size)
    image_binary =  io.BytesIO()
    image.save(image_binary, format='jpeg')
    image_file.seek(0)

    #Send uploaded image to server
    if reply == True:
        #Send image to server             
        print "Sending image to server"
        res = requests.post(
            UPLOAD_URL,
            files={'image': ('image.jpg', image_binary)}
        )
        res.raise_for_status()
        res_json = res.json()

        #Diabetic retinopathy prediction
        pred = res_json.pop("pred", None)

        #Collect processed images for display
        print "Obtaining model results..."
        image_dict = {}
        for im in res_json:
            image_url = REQUEST_IMAGE_URL + "/{}".format(res_json[im])

            res = requests.get(image_url)
            res.raise_for_status()
            im_data = Image.open(io.BytesIO(res.content))

            image_dict[im] = im_data
            time.sleep(0.1)


        #Display results to user
        fig = plt.figure()

        plt.subplot(221)
        plt.imshow(image)
        plt.title('Original Image')

        plt.subplot(222)
        plt.imshow(image_dict["im_p"])
        plt.title("Processed Image")

        plt.subplot(223)
        plt.imshow(image_dict["hm"])
        plt.title("Heatmap")

        plt.subplot(224)
        plt.imshow(image_dict["hm_im"])
        plt.title("Heatmap Superimposed Image")

        plt.show()


except:
	raise
finally:
    # camera.close()
    pass
