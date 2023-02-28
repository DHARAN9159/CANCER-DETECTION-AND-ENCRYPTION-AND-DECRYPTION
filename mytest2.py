import matplotlib.pyplot as plt,numpy as np,random
from numpy import zeros
#Enter the address of the image inside r""
#Enter a jpg image.
img=plt.imread(r"im1.png")

rows=img.shape[0]
col=img.shape[1]

plt.title("Original Image")
plt.imshow(img)
plt.show()

key=[]
while len(key)!= 256:
    j=random.randrange(0,256)
    if j not in key:
        key.append(j)
print(key)

def encrypt_image(img,key,rows,col):
    img1=np.zeros((rows,col,3),dtype=int)
    for i in range(rows):
        for j in range(col):
            for k in range(3):
                img1[i][j][k]=key[img[i][j][k]]
    return img1

img1=encrypt_image(img,key,rows,col)
plt.title("Encrypted Image")
plt.imshow(img1)
plt.show()




def decrypt_image(img,key,rows,col):
    img1=np.zeros((rows,col,3),dtype=int)
    for i in range(rows):
        for j in range(col):
            for k in range(3):
                img1[i][j][k]=key.index(img[i][j][k])
    return img1

img3=decrypt_image(img1,key,rows,col)
plt.title("Decrypted Image")
plt.imshow(img3)
plt.show()