import os
import cv2

src = "./src_image/"
des = "./des_image/"
fs = os.listdir(src)

for f1 in fs:
    img_path = os.path.join(src, f1)
    print(img_path)
    img = cv2.imread(img_path)
    (h, w, a) = img.shape
    # calculate the center of the image
    center = (w / 2, h / 2)
    angle90 = 90
    scale = 1.0
    M = cv2.getRotationMatrix2D(center, angle90, scale)
    rotated90 = cv2.warpAffine(img, M, (h, w))
    des_path = os.path.join(des, f1)
    cv2.imwrite(des_path, rotated90)
    