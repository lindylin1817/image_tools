import cv2
import numpy as np
from PIL import Image
import os

images_dir = "../siamese-triplet/myevaldata/"
crop_x1 = 1200
crop_y1 = 0
crop_x2 = 3400
crop_y2 = 1800

crop_array = (crop_x1, crop_y1, crop_x2, crop_y2)

fs = os.listdir(images_dir)
for f1 in fs:
    img_path = os.path.join(images_dir, f1)
    if os.path.isdir(img_path):
        dir_path = img_path
        for f2 in os.listdir(dir_path):
            img_path = os.path.join(dir_path, f2)
            img = Image.open(img_path)
            if img.size[0] < img.size[1]:
                img = img.transpose(Image.ROTATE_270)
            img_crop = img.crop(crop_array)
            img_crop.save(img_path)
            print(img_path)
    else:
        img = Image.open(img_path)
        if img.size[0] < img.size[1]:
            img = img.transpose(Image.ROTATE_270)
        img_crop = img.crop(crop_array)
        img_crop.save(img_path)
        print(img_path)