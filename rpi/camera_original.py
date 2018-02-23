import picamera
import time
import RPi.GPIO as GPIO

GPIO.cleanup()
GPIO.setmode(GPIO.BCM)
GPIO.setup(17, GPIO.IN, GPIO.PUD_UP)
GPIO.setup(22, GPIO.OUT)
GPIO.setup(23, GPIO.OUT)
GPIO.output(22, GPIO.HIGH)
GPIO.output(23, GPIO.HIGH)

with picamera.PiCamera() as camera:
    camera.start_preview()
    GPIO.wait_for_edge(17, GPIO.FALLING)
    GPIO.output(22, GPIO.LOW)
    GPIO.output(23, GPIO.HIGH)
    print('led off?')
    camera.capture('/home/pi/Desktop')
    camera.stop_preview()
    GPIO.cleanup()
