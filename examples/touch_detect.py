# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import pytouch
from pytouch.handlers import ImageHandler
from pytouch.sensors import DigitSensor
from pytouch.tasks import TouchDetect


def touch_detect_notouch(index):
    source = ImageHandler('/home/acldifstudent2/Code/Jiayun/PyTouch/img_touch/test/class0/img' + str(index) + '.png')

    # initialize with task defaults
    pt = pytouch.PyTouch(DigitSensor, tasks=[TouchDetect])
    is_touching, certainty = pt.TouchDetect(source.image)

    # initialize with custom configuration of TouchDetect task
    # touch_detect = TouchDetect(DigitSensor, zoo_model="touchdetect_resnet18")

    touch_detect = TouchDetect(DigitSensor, model_path= "/home/acldifstudent2/Code/Jiayun/PyTouch/train/outputs/2022-07-29/11-51-24/checkpoints/touch_detect_exp-epoch=4_val_loss=54.635_val_acc=0.538.ckpt")

    is_touching, certainty = touch_detect(source.image)

    print(f"Is touching? {is_touching}, {certainty}")

    return is_touching


def touch_detect_touch(index):
    source = ImageHandler('/home/acldifstudent2/Code/Jiayun/PyTouch/img_touch/test/class1/img' + str(index) + '.png')

    # initialize with task defaults
    pt = pytouch.PyTouch(DigitSensor, tasks=[TouchDetect])
    is_touching, certainty = pt.TouchDetect(source.image)

    # initialize with custom configuration of TouchDetect task
    # touch_detect = TouchDetect(DigitSensor, zoo_model="touchdetect_resnet18")

    touch_detect = TouchDetect(DigitSensor, model_path= "/home/acldifstudent2/Code/Jiayun/PyTouch/train/outputs/2022-07-29/11-51-24/checkpoints/touch_detect_exp-epoch=4_val_loss=54.635_val_acc=0.538.ckpt")

    is_touching, certainty = touch_detect(source.image)

    print(f"Is touching? {is_touching}, {certainty}")

    return is_touching


if __name__ == "__main__":
    print("notouch:")
    count_notouch = 0
    for i in range(1, 21):
        print(i)
        temp = touch_detect_notouch(i)
        if temp == 0:
            count_notouch = count_notouch + 1

    print("touch:")
    count_touch = 0
    for i in range(1, 21):
        print(i)
        temp = touch_detect_touch(i)
        if temp == 1:
            count_touch = count_touch + 1

    print("Accuracy for no touch:", count_notouch/20*100, "%")
    print("Accuracy for touch:", count_touch / 20*100, "%")
