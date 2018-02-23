import io
import time
import picamera
import requests
import PIL
from PIL import Image
from easygui import ccbox
#import RPi.GPIO as GPIO
import tempfile
import IPython

#Set RaspPi pins
#GPIO.setmode(GPIO.BCM)
#GPIO.setup(17, GPIO.IN, GPIO.PUD_UP)
#GPIO.setup([22, 23], GPIO.OUT)
#GPIO.output([22, 23], GPIO.HIGH)
    
#Define API_URL
API_URL = "http://160.39.187.57:5000/upload"

try:
    camera = picamera.PiCamera()
    camera.resolution = (512, 512)
    camera.start_preview()
    stream = io.BytesIO()
    
    #Wait for shutter press
#    GPIO.wait_for_edge(17, GPIO.FALLING)

	#Pause before capturing
    time.sleep(2)
    #GPIO.output([22, 23], [GPIO.LOW, GPIO.HIGH])

    camera.capture(stream, 'jpeg')
    camera.stop_preview()
    #GUI to interact with server
    tf = tempfile.NamedTemporaryFile()
    image = Image.open(stream)
    #image.thumbnail((512, 512), PIL.Image.ANTIALIAS)
    
    image.save(tf, format='jpeg')
    
    reply = ccbox("Do you want to analyze this image?", "Image Viewer", 
        image=tf.name)

    if reply == True:
        with open(tf.name, "r") as f:                
            print "Sending image to server"
            requests.post(
                API_URL,
                files={'image': ('image.jpg', f)}
            )

            print("Request sent to server")
except:
	raise
finally:
    camera.close()
