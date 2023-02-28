import cv2
import warnings
import matplotlib.pyplot as plt
import numpy as np
# warnings.filterwarnings('ignore')
# # import cv2.imshow
# # from google.colab.patches import cv2_imshow
# img=cv2.imread('data/brain_tumor_dataset/yes/Y29.jpg')
# cv2.imshow("imager",img)
# b,g,r= cv2.split("image",img)
# print(cv2.imshow(b))
# cv2.imshow(g)
# cv2.imshow(r)
#


import cv2
# img = cv2.imread('data/brain_tumor_dataset/yes/Y29.jpg')
# imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow("Gray Image", imgGray)
# cv2.waitKey(0)


import cv2
import numpy as np
from matplotlib import pyplot as plt
# img = cv2.imread('data/brain_tumor_dataset/yes/Y29.jpg')
# kernel = np.ones((5,5),np.float32)/25
# dst = cv2.filter2D(img,-1,kernel)
# plt.subplot(121),plt.imshow(img),plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(dst),plt.title('Filter2D')
# plt.xticks([]), plt.yticks([])
# plt.show()

#
# from typing import no_type_check
# import cv2
# # from google.colab.patches import cv2_imshow
# image = cv2.imread(r'data/brain_tumor_dataset/yes/Y29.jpg',1)
# cv2.imshow("image is ",image)
# plt.show()
# (b,g,r) = image[50,50]
# print(r,g,b)
# image[50,50] = (0,255,255)
# print(r,g,b)
# 
# cv2.imshow("image is",image)
# plt.show()
# cv2.waitKey(0)
# img="aaa"
# # from google.colab.patches importcv2_imshow
# blurImg=cv2.blur(,(20,20))
# cv2.imshow(blurImg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()