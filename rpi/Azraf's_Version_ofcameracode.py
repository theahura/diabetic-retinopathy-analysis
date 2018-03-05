import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import picamera
import RPi.GPIO as GPIO
import io

import tempfile
import time

from PIL import Image
from easygui import ccbox
import requests

# import os
import argparse


    
#Define api urls
SERV_ADDR = "http://drcam.molabs.tech"
UPLOAD_URL = SERV_ADDR + "/upload"
REQUEST_IMAGE_URL = SERV_ADDR + "/image"


def main():
    parser = argparse.ArgumentParser(description="Take and process retinal images")
    parser.add_argument("--auto", action="store_true")

    args = parser.parse_args()
    #Decide if image should be taken after a delay
    auto = args.auto

    if not auto:
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(17, GPIO.IN, GPIO.PUD_DOWN)
        GPIO.setwarnings(False)
        GPIO.setup(18,GPIO.OUT)
        GPIO.setup(15, GPIO.OUT)

        GPIO.output(15, GPIO.LOW)
        GPIO.output(18,GPIO.LOW)



    camera = picamera.PiCamera()
    try:
        #Load picamera and begin preview
        camera.resolution = (512, 512)
        stream = io.BytesIO()

        camera.start_preview()
        if not auto:
            #Wait for shutter press
            GPIO.wait_for_edge(17, GPIO.FALLING)
            GPIO.output(15, GPIO.HIGH)
            GPIO.output(18, GPIO.LOW)
            
        else:
            #Delay
            time.sleep(2)
        
        camera.capture(stream, 'jpeg')
        GPIO.output(18, GPIO.HIGH)
        camera.stop_preview()

         

        #Process image and save in temp file
        image = Image.open(stream)
        # image = Image.open(os.path.expanduser("~/Downloads/217_left.jpeg"))
        orig_size = image.size
        image.thumbnail((512, 512), Image.ANTIALIAS)
        tf = tempfile.NamedTemporaryFile()
        image.save(tf, format='jpeg')

        #Display selection GUI
        print("Loading GUI")
        reply = ccbox("Do you want to analyze this image?", "Image Viewer", 
            image=tf.name)

        #Resize image back to original size for server
        image = image.resize(orig_size)
        image_binary =  io.BytesIO()
        image.save(image_binary, format='jpeg')
        image_binary.seek(0)

        #Send uploaded image to server
        if reply == True:
            #Send image to server             
            print("Sending image to server")
            res = requests.post(
                UPLOAD_URL,
                files={'image': ('image.jpg', image_binary)}
            )
            res.raise_for_status()
            res_json = res.json()

            #Diabetic retinopathy prediction
            pred = res_json.pop("pred", None)

            #Collect processed images for display
            print("Obtaining model results...")
            image_dict = {}
            for im in res_json:
                image_url = REQUEST_IMAGE_URL + "/{}".format(res_json[im])

                res = requests.get(image_url)
                res.raise_for_status()
                im_data = Image.open(io.BytesIO(res.content))

                image_dict[im] = im_data
                time.sleep(0.1)


            #Display results to user
            fig = plt.figure(figsize=(10,8))

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

            plt.suptitle('Diabetic Retinopathy Level: {}'.format(pred))
            plt.show()

    except:
        raise
    finally:
        if not auto:
            GPIO.cleanup()
        camera.close()
        print("exiting")    

if __name__ == "__main__":
    main()



