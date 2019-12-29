import cv2
import numpy as np
from PIL import Image

img = Image.open("10044.jpg")
if img.size[0] < img.size[1]:
    img = img.transpose(Image.ROTATE_270)
img.save("./org_img.jpg")
img = cv2.imread("./org_img.jpg")

w, h, a = img.shape
max_radius = round(h/2) + 300
min_radius = max_radius - 400
print(max_radius, min_radius)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('gray.jpg', gray)
blur = cv2.blur(gray, (15, 15))
cv2.imwrite('blur.jpg', blur)

canny = cv2.Canny(blur, 50, 100)
cv2.imwrite("canny.jpg", canny)
print("aaaa")
circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 100, param1=30, param2=50, minRadius=min_radius,
                           maxRadius=max_radius)
print(circles)
print(np.max(circles[0][:,2]))


print(len(circles[0]))
i = 0
for circle in circles[0]:

    print(circle[2])
    # 坐标行列(就是圆心)
    x = int(circle[0])
    y = int(circle[1])
    # 半径
    r = int(circle[2])
    # 在原图用指定颜色圈出圆，参数设定为int所以圈画存在误差
    img = cv2.circle(gray, (x, y), r, (0, 0, 255), 3, 8, 0)
    filename = "./out_" + str(i) + ".jpg"
    cv2.imwrite(filename, img)

    i = i + 1


