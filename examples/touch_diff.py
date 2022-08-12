import pytouch
from pytouch.handlers import ImageHandler
from pytouch.sensors import DigitSensor
from pytouch.tasks import TouchDetect

import digit_interface
from digit_interface.digit import Digit

import numpy as np
import cv2
import os


def touch_or_not(imagepath_compare, directory_name, current_class):
    count = 0
    img_compare = cv2.imread(imagepath_compare)
    img_compare = cv2.resize(img_compare, (224, 224))

    for filename in os.listdir(directory_name):
        # print(filename)
        img = cv2.imread(directory_name + "/" + filename)
        img = cv2.resize(img, (224, 224))

        diff = img_compare.astype(np.float32) - img.astype(np.float32)
        result = np.sum(abs(diff) > 37)
        # print(result)
        if current_class == 0:
            if result < 160:
                # print("not touch")
                count = count + 1
            else:
                # print("touch")
                pass
        else:
            if result < 160:
                pass
                # print("not touch")
            else:
                # print("touch")
                count = count + 1

    return count


if __name__ == "__main__":
    imagePath_compare = "/home/acldifstudent2/Code/Jiayun/PyTouch/img_anothersensor/train/compare_img.png"

    print("Not touch:")
    no_touch = 0
    count_notouch = touch_or_not(imagePath_compare, "/home/acldifstudent2/Code/Jiayun/PyTouch/img_anothersensor/train/class0", no_touch)
    print("Accuracy for no touch:", count_notouch / 500 * 100, "%")

    print("Touch:")
    touch = 1
    count_touch = touch_or_not(imagePath_compare, "/home/acldifstudent2/Code/Jiayun/PyTouch/img_anothersensor/train/class1", touch)
    print("Accuracy for touch:", count_touch / 500 * 100, "%")
