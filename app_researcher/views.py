from django.shortcuts import render,redirect
from . models import *
from app_user. models import *
import os
from django.conf import settings
from django.http import HttpResponse, Http404
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import warnings
import cv2 as cv
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


from keras.preprocessing.image import ImageDataGenerator
from keras.applications import vgg16
from keras.models import Model
from keras.layers import Dense, MaxPool2D, Conv2D
import keras
import seaborn as sns
import pandas as pd
import tensorflow as tf
from keras.models import load_model
import keras.utils as image
import matplotlib.cm as cm

from IPython.display import Image, display

import matplotlib.pyplot as plt


from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

from keras.preprocessing.image import ImageDataGenerator
from keras.applications import vgg16
from keras.models import Model
from keras.layers import Dense, MaxPool2D, Conv2D
import keras
import seaborn as sns
import pandas as pd
import tensorflow as tf
from keras.models import load_model
import keras.utils as image
import matplotlib.cm as cm
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import vgg16
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import cv2
import math
import matplotlib.pyplot as plt

import os
import pandas as pd
import scipy.ndimage
from skimage import measure, morphology
from skimage.transform import rotate
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import tensorflow as tf


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from django.conf import settings

from django.contrib import messages

# Create your views here.
def researcherregister(request):
    if request.method=="POST":
        name = request.POST['name']
        email =request.POST['email']
        phone_number = request.POST['phone_number']
        password =request.POST['password']
        confirm_password =request.POST['confirm_password']
        researcher_regilog(name=name,email=email,password=password,phone_number=phone_number,
              confirm_password=confirm_password ).save()
        messages.info(request, 'registered sucessfully.')


    return render(request,'Research/researcher_regis.html')


def researcherlogin(request):
    if request.method=="POST":
        email=request.POST['email']
        password=request.POST['password']
        try:
            te = researcher_regilog.objects.get(email=email, password=password)
            messages.info(request, 'Sucessfully login')
            return redirect('/researchhome/')

        except:
            return redirect('/researcher_logi/')

    return render(request,'Research/researcher_login.html')


def researchhome(request):
    messages.info(request, 'Sucessfully login')
    return render(request,'Research/research_home.html')


def blur(request):
    data = upload.objects.all()
    return render (request,'Research/blur.html',{'data':data})
def blur1(request):
    data = upload.objects.all()
    return render(request,'Research/blur.html',{'data':data})

def blurimage(request,id):
    import cv2 as cv
    import numpy as np
    from matplotlib import pyplot as plt
    data = upload.objects.get(id=id)
    img = data.image
    imgpath = f'{settings.MEDIA_ROOT}/{img}'
    img = cv.imread(imgpath)
    blur1 = cv.blur(img, (5, 5))
    plt.subplot(121), plt.imshow(img), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(blur1), plt.title('Blurred')
    plt.xticks([]), plt.yticks([])
    plt.show()
    cv.waitKey(0)
    return redirect('/researchhome/')

def filter(request):
    data=upload.objects.all()
    return  render (request,"Research/filter.html",{'data':data})

def filterimage(request,id):
    import cv2
    import numpy as np
    from matplotlib import pyplot as plt
    data = upload.objects.get(id=id)
    img = data.image
    print("hi")
    imgpath = f'{settings.MEDIA_ROOT}/{img}'
    img = cv2.imread(imgpath)
    kernel = np.ones((5,5),np.float32)/25
    dst = cv2.filter2D(img,-1,kernel)
    plt.subplot(121),plt.imshow(img),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(dst),plt.title('Filter2D')
    plt.xticks([]), plt.yticks([])
    plt.show()
    cv2.waitkey(0)
    return redirect('/researchhome/')

def grayscale(request):
    data = upload.objects.all()
    return render (request,'Research/grayscale.html',{'data':data})

def grayscaleimage(request,id):
    import cv2
    from matplotlib import pyplot as plt
    data = upload.objects.get(id=id)
    img = data.image
    imgpath = f'{settings.MEDIA_ROOT}/{img}'
    img = cv2.imread(imgpath)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gray Image", imgGray)
    plt.show()
    cv2.waitKey(0)
    return redirect('/researchhome/')

def resize(request):
    data = upload.objects.all()

    return render(request,'Research/resize.html',{'data':data})

def resizeimage(request,id):
    from numpy import expand_dims
    from tensorflow.keras.utils import load_img
    from tensorflow.keras.utils import img_to_array
    from keras.preprocessing.image import ImageDataGenerator
    from matplotlib import pyplot
    # load the image
    data = upload.objects.get(id=id)
    img = data.image
    imgpath = f'{settings.MEDIA_ROOT}/{img}'
    img = load_img(imgpath)
    # convert to numpy array
    data = img_to_array(img)

    # expand dimension to one sample
    samples = expand_dims(data, 0)
    # create image data augmentation generator
    datagen = ImageDataGenerator(width_shift_range=[-200, 200])
    # prepare iterator
    it = datagen.flow(samples, batch_size=1)
    # generate samples and plot
    for i in range(9):
        # define subplot
        pyplot.subplot(330 + 1 + i)
        # generate batch of images
        batch = it.next()
        # convert to unsigned integers for viewing
        image = batch[0].astype('uint8')
        # plot raw pixel data
        pyplot.imshow(image)
    # show the figure
    pyplot.show()
    return redirect('/researchhome/')


def download(request, id):
    data = upload.objects.get(id=id)
    img = data.image
    file_path = f'{settings.MEDIA_ROOT}/{img}'
    if os.path.exists(file_path):
        with open(file_path, 'rb') as fh:
            response = HttpResponse(fh.read(), content_type="application/vnd.ms-excel")
            response['Content-Disposition'] = 'inline; filename=' + os.path.basename(file_path)
            messages.info(request, "successfully image Downloaded")
            return response
    raise Http404


def predictionimage(request):
    data = upload.objects.all()
    return render(request,'Research/predictimage/predictionimage.html',{'data':data})

def delete(request,id):
    y = upload.objects.get(id=id)
    y.delete()
    return redirect('/predictionimage/')


def analyse_image(request, id):
    b={0:"Cancer",1:"No Cancer"}
    data=upload.objects.get(id=id)
    img=data.image
    imgpath = f'{settings.MEDIA_ROOT}/{img}'

    print(imgpath)
    model = load_model("modelcheck/bestmodel.h5")

    path = imgpath
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = x / 255
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)
    global a
    a =preds
    v=a[0]
    u=b[int(v)]
    print(u)

    r = data.id
    st = upload.objects.filter(id=r).update(result=u)

    return redirect('/researchhome/')







    # b = plt.imread(data1)
    # plt.imshow(b, cmap="viridis")
    # plt.title("Original image")
    # plt.show()
    # print(plt.show())

    # if a == 0:
    #     x="cancer"
        # print(x)

    # else:
    #     # x="No cancer"
    #     data1 = upload.objects.get(id=id)
    #     r = data1.id
    #     st = upload.objects.filter(id=r).update(result=a)
    #     # print(x)


    # b = plt.imread(path)
    # plt.imshow(b, cmap="viridis")
    # plt.title("Original image")
    # plt.show()
    # print(plt.show())


# def show(self,analyse_image):
#
#     b = plt.imread(self.path)
#     plt.imshow(b, cmap="viridis")
#     plt.title("Original image")
#     plt.show()
#     print(plt.show())
# show()


def resultimage(request):
    data = upload.objects.all()
    return render(request,'Research/result/resultimage.html',{'data':data})


# def show(request):
#     import cv2 as cv
#     import matplotlib.pyplot as plt
#     data1 = upload.objects.get(id=id)
#
#
#     img = cv.imread()  # reading the image
#     ret, thresh1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
#     plt.imshow(thresh1, cmap="viridis")
#     plt.show()

def show1(request):
    data = upload.objects.all()
    return render(request,'Research/result/resultimage.html',{'data': data})


def show(request,id):
    data = upload.objects.get(id=id)
    # img = data.image
    # imgpath = f'{settings.MEDIA_ROOT}/{img}'
    # imge = cv.imread(imgpath)
    # ret,thresh1 = cv.threshold(imge,127, 255, cv.THRESH_BINARY)
    # plt.imshow(thresh1, cmap="viridis")
    # plt.show()

    img = data.image
    imgpath = f'{settings.MEDIA_ROOT}/{img}'
    img = cv2.imread(imgpath)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gray Image", imgGray)
    plt.show()
    cv2.waitKey(0)

    return redirect('/show1/')
    # except show.DoesnotExist:
    #     raise Http404("Does not exist")



