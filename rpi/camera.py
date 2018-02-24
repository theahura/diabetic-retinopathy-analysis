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


    
#Define API_URL
# API_URL = "http://54.164.115.173:80/upload"
SERV_ADDR = "http://drcam.molabs.tech"
UPLOAD_URL = SERV_ADDR + "/upload"
REQUEST_IMAGE_URL = SERV_ADDR + "/image"

try:
 #    camera = picamera.PiCamera()
 #    camera.resolution = (512, 512)
 #    camera.start_preview()
 #    stream = io.BytesIO()
    
 #    #Wait for shutter press

	# #Pause before capturing
 #    time.sleep(2)

 #    camera.capture(stream, 'jpeg')
 #    camera.stop_preview()
 #    #GUI to interact with server
    image = Image.open(os.path.expanduser("~/Downloads/217_left.jpeg"))
    orig_size = image.size

    image.thumbnail((512, 512), PIL.Image.ANTIALIAS)

    tf = tempfile.NamedTemporaryFile()
 #    image = Image.open(stream)
    
    image.save(tf, format='jpeg')

    
    print "Loading GUI"
    # IPython.embed()
    reply = ccbox("Do you want to analyze this image?", "Image Viewer", 
        image=tf.name)
    print "GUI Loaded"

    image = image.resize(orig_size)

    image_file =  io.BytesIO()
    image.save(image_file, format='jpeg')
    image_file.seek(0)

    if reply == True:
        with open(tf.name, "r") as f:   
            #Send image to server             
            print "Sending image to server"
            res = requests.post(
                UPLOAD_URL,
                files={'image': ('image.jpg', image_file)}
            )
            res.raise_for_status()
            #Parse response
            res_json = res.json()
            pred = res_json.pop("pred", None)

            image_dict = {}
            for im in res_json:
                image_url = REQUEST_IMAGE_URL + "/{}".format(res_json[im])

                res = requests.get(image_url)
                im_data = Image.open(io.BytesIO(res.content))

                image_dict[im] = im_data
                time.sleep(0.3)

            fig = plt.figure()

            plt.subplot(221)
            plt.imshow(image)
            plt.title('Original')

            plt.subplot(222)
            plt.imshow(image_dict["im_p"])
            plt.title("Im P")

            plt.subplot(223)
            plt.imshow(image_dict["hm"])
            plt.title("Heatmap")

            plt.subplot(224)
            plt.imshow(image_dict["hm_im"])
            plt.title("Headmap Superimposed Image")


            plt.show()


            print("Request sent to server")
except:
	raise
finally:
    # camera.close()
    pass
