from PIL import Image
from utils import CLR2Gray

src_path = "./10044.jpg"
dst_path = "./gray_11_59_30.jpg"
dst2_path = "./gray_69_22_9.jpg"
dst3_path = "./gray_1_1_98.jpg"

src_img = Image.open(src_path)

rgb_array = [0.11, 0.59, 0.30]
dst_img = CLR2Gray(src_img, rgb_array)
dst_img.save(dst_path)

rgb_array = [0.69, 0.22, 0.09]
dst_img = CLR2Gray(src_img, rgb_array)
dst_img.save(dst2_path)

rgb_array = [0.01, 0.01, 0.98]
dst_img = CLR2Gray(src_img, rgb_array)
dst_img.save(dst3_path)
