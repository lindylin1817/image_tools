import cv2
import numpy as np
from PIL import Image
import os

images_dir = "./data/"
aug_dir = "./data_aug/"
fs = os.listdir(images_dir)
for f1 in fs:
    img_path = os.path.join(images_dir, f1)
    if os.path.isdir(img_path):
        dir_path = img_path
        ver_dir = aug_dir + f1 + "/"
        os.mkdir(ver_dir)
        for f2 in os.listdir(dir_path):
            img_path = os.path.join(dir_path, f2)
            img = cv2.imread(img_path)
            xImg = cv2.flip(img, 1, dst=None)  # 水平镜像
            xImg1 = cv2.flip(img, 0, dst=None)  # 垂直镜像
            xImg2 = cv2.flip(img, -1, dst=None)  # 对角镜像
            i = len(images_dir)
            tmp_path = ver_dir + f2
            new_img_path = tmp_path[:tmp_path.rindex(".")] + "_horizon.jpg"
            print(new_img_path)
            cv2.imwrite(new_img_path, xImg)
            new_img_path = tmp_path[:tmp_path.rindex(".")] + "_vertical.jpg"
            cv2.imwrite(new_img_path, xImg1)
            new_img_path = tmp_path[:tmp_path.rindex(".")] + "_corner.jpg"
            cv2.imwrite(new_img_path, xImg2)
