import cv2 as cv
import numpy as np
from picamera import PiCamera
from picamera.array import PiRGBArray
from time import sleep


def load_cascade(f_cascade):
    return cv.CascadeClassifier(f_cascade)


def main(f_cascade):
    face_cascade = load_cascade(f_cascade)
    with PiCamera(resolution=(816, 480), framerate=12) as camera:
        camera.rotation = 180
        camera.start_preview()
        sleep(3)
        with PiRGBArray(camera) as output:
            camera.capture(output, "bgr")
            image = output.array
        #camera.stop_preview()
    grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grey, 1.1, 5)
    for x, y, w, h in faces:
        cv.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)
    cv.imshow("faces?", image)
    cv.waitKey()
    cv.destroyAllWindows()
    

F_CASCADE = "cascades/haarcascade_frontalface_default.xml"

if __name__ == "__main__":
    main(F_CASCADE)
