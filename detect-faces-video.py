import cv2 as cv
import numpy as np
from picamera import PiCamera
from picamera.array import PiRGBArray
from time import sleep
from typing import Tuple


def check_key_press() -> bool:
    return cv.waitKey(1) & 0xFF == ord("q")


def draw_rectangles(image: np.array, rectangles: Tuple[int]):
    for x, y, w, h in rectangles:
        cv.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)


def find_faces(face_cascade: cv.CascadeClassifier, image: np.array) -> Tuple[int]:
    return face_cascade.detectMultiScale(image_to_grey(image), 1.1, 5)


def image_to_grey(image: np.array) -> np.array:
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)


def let_camera_warm_up():
    sleep(2)


def load_cascade(f_cascade: str) -> cv.CascadeClassifier:
    return cv.CascadeClassifier(f_cascade)


def process_and_display(face_cascade: cv.CascadeClassifier, image: np.array):
    faces = find_faces(face_cascade, image)
    draw_rectangles(image, faces)
    cv.imshow("frame", image)


def rotate(camera: PiCamera, rotation: int=180):
    camera.rotation = rotation


def main(face_cascade: cv.CascadeClassifier, resolution: Tuple[int], framerate: int):
    with PiCamera(resolution=resolution, framerate=framerate) as camera:
        rotate(camera)
        let_camera_warm_up()
        with PiRGBArray(camera) as output:
            for frame in camera.capture_continuous(output, format="bgr", use_video_port=True):
                process_and_display(face_cascade, output.array)               
                output.truncate(0)
                if check_key_press():
                    cv.destroyAllWindows()
                    break


F_CASCADE = "cascades/haarcascade_frontalface_default.xml"
RESOLUTION = (640, 480)
FRAMERATE = 12

if __name__ == "__main__":
    main(load_cascade(F_CASCADE), RESOLUTION, FRAMERATE)
