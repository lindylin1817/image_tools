import cv2
import numpy as np
from PIL import Image

img = Image.open("./data/10043.jpg")

if img.size[0] < img.size[1]:
    img = img.transpose(Image.ROTATE_270)
w = img.size[0]
h = img.size[1]
img.save("./org_img.jpg")
print(w,", ", h)

img = cv2.imread("./org_img.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('gray.jpg', gray)
#blur = cv2.blur(gray, (15, 15))
#cv2.imwrite('blur.jpg', blur)
#canny = cv2.Canny(gray, 50, 100)
#cv2.imwrite("canny.jpg", canny)

ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
cv2.imwrite("binary.jpg", binary)

pix_count_array = np.zeros(w, dtype=int)
for w_i in range (0, w):
    pix_count_array[w_i] = np.sum(binary[:, w_i])
max_pix_count = np.max(pix_count_array)
print(max_pix_count)
top_k_idx=pix_count_array.argsort()[::-1][0:200]
print(top_k_idx)
mean_center = np.mean(top_k_idx)
print(mean_center)
for i in range(0, w):
    if pix_count_array[i] == max_pix_count:
        center = i
        print("center is ", center)
        break
for i in range(0, w):
    print(i, " : ", pix_count_array[i])

