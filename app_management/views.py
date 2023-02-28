from django.shortcuts import render, redirect
from django.contrib import messages
from django.core.mail import send_mail
from .models import *
from app_user.models import *
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from django.conf import settings
import matplotlib.pyplot as plt
import cv2 as cv
from keras.models import load_model
import cv2
import math
import matplotlib.pyplot as plt
import keras.utils as image
import os
from django.core.mail import EmailMessage
from django.conf import settings
from django.template.loader import render_to_string
from django.utils.html import strip_tags
from PIL import Image

from django.core.mail import EmailMultiAlternatives

# dec enc
from PIL import Image
import numpy as np
import os
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import cv2
import random
from math import log

from tqdm import tqdm


# Create your views here.
def adminlogin(request):
    if request.method == "POST":
        email = request.POST["email"]
        password = request.POST["password"]
        print(email)
        if email == "admin@gmail.com" and password == "admin":
            print(email)
            request.session['admin'] = "admin@gmail.com"
            messages.info(request, "Successfully Login ")
            return render(request, 'Admin/admin_home.html')
        elif email != "admin@gmail.com":
            # messages.error(request, "Wrong Mail id")
            return render(request, 'admin/admin.html')
        elif password != "admin":
            # messages.error(request,"wrong password")
            return render(request, 'admin/admin.html')
        else:
            return render(request, 'admin/admin.html')
    return render(request, 'Admin/admin.html')


def admin_home(request):
    return render(request, 'Admin/admin_home.html')


def dashboard1(request):
    data = basic_details.objects.all()
    return render(request, 'Admin/Dash1/Dash1.html', {'data': data})


def dashboard2(request):
    data = upload.objects.all()

    return render(request, 'Admin/Dash2/dashboard2.html', {'data': data})


def showimage1(request, id):
    data = upload.objects.get(id=id)
    img = data.image
    # imgpath = f'{settings.MEDIA_ROOT}/{img}'
    # imge = cv.imread(imgpath)  # reading the image
    # ret, thresh1 = cv.threshold(imge, 127, 255, cv.THRESH_BINARY)
    # plt.imshow(thresh1,cmap="viridis")
    # plt.show()
    # plt.waitkey(0)
    imgpath = f'{settings.MEDIA_ROOT}/{img}'
    img = cv2.imread(imgpath)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gray Image", imgGray)
    plt.show()
    cv2.waitKey(0)
    return redirect('/adminhome/')


def showresult(request, id):
    data = upload.objects.get(id=id)
    res = data.result
    if res == "Cancer":
        return redirect('/cancer/')
    else:
        return redirect('/normal/')

    return render(request, 'Admin/Dash2/dashboard2.html')


def cancer(request):
    return render(request, 'Admin/result/cancer.html')


def normal(request):
    return render(request, 'Admin/result/normal.html')


def encrypt(request):
    data = upload.objects.all()

    return render(request, "Admin/encrypt/encypt.html", {'data': data})


def encry(request):
    data = upload.objects.all()
    return render(request, "Admin/generate/encry.html", {'data': data})


# def generate(request,id):
#
#     import numpy as np
#     a=np.random.choice(250)
#     b=upload.objects.all()
#     b.key=a
#     b.save()
#
#     return redirect('/encrypt/')
#
def encryptimage(request, id):
    data = upload.objects.get(id=id)
    r = data.id
    print(data)
    img = data.image
    print("hi")
    path = f'{settings.MEDIA_ROOT}/{img}'

    def getImageMatrix(imageName):
        im = Image.open(imageName)
        pix = im.load()
        color = 1
        if type(pix[0, 0]) == int:
            color = 0
        image_size = im.size
        image_matrix = []
        for width in range(int(image_size[0])):
            row = []
            for height in range(int(image_size[1])):
                row.append((pix[width, height]))
            image_matrix.append(row)

        return image_matrix, image_size[0], image_size[1], color

    def LogisticEncryption(imageName, key):
        N = 256
        key_list = [ord(x) for x in key]
        G = [key_list[0:4], key_list[4:8], key_list[8:12]]
        g = []
        R = 1
        for i in range(1, 4):
            s = 0
            for j in range(1, 5):
                s += G[i - 1][j - 1] * (10 ** (-j))
            g.append(s)
            R = (R * s) % 1

        L = (R + key_list[12] / 256) % 1
        S_x = round(((g[0] + g[1] + g[2]) * (10 ** 4) + L * (10 ** 4)) % 256)
        V1 = sum(key_list)
        V2 = key_list[0]
        for i in range(1, 13):
            V2 = V2 ^ key_list[i]
        V = V2 / V1

        L_y = (V + key_list[12] / 256) % 1
        S_y = round((V + V2 + L_y * 10 ** 4) % 256)
        C1_0 = S_x
        C2_0 = S_y
        C = round((L * L_y * 10 ** 4) % 256)
        C_r = round((L * L_y * 10 ** 4) % 256)
        C_g = round((L * L_y * 10 ** 4) % 256)
        C_b = round((L * L_y * 10 ** 4) % 256)
        x = 4 * (S_x) * (1 - S_x)
        y = 4 * (S_y) * (1 - S_y)

        imageMatrix, dimensionX, dimensionY, color = getImageMatrix(imageName)
        LogisticEncryptionIm = []
        for i in range(dimensionX):
            row = []
            for j in range(dimensionY):
                while x < 0.8 and x > 0.2:
                    x = 4 * x * (1 - x)
                while y < 0.8 and y > 0.2:
                    y = 4 * y * (1 - y)
                x_round = round((x * (10 ** 4)) % 256)
                y_round = round((y * (10 ** 4)) % 256)
                C1 = x_round ^ ((key_list[0] + x_round) % N) ^ ((C1_0 + key_list[1]) % N)
                C2 = x_round ^ ((key_list[2] + y_round) % N) ^ ((C2_0 + key_list[3]) % N)
                if color:
                    C_r = ((key_list[4] + C1) % N) ^ ((key_list[5] + C2) % N) ^ (
                                (key_list[6] + imageMatrix[i][j][0]) % N) ^ ((C_r + key_list[7]) % N)
                    C_g = ((key_list[4] + C1) % N) ^ ((key_list[5] + C2) % N) ^ (
                                (key_list[6] + imageMatrix[i][j][1]) % N) ^ ((C_g + key_list[7]) % N)
                    C_b = ((key_list[4] + C1) % N) ^ ((key_list[5] + C2) % N) ^ (
                                (key_list[6] + imageMatrix[i][j][2]) % N) ^ ((C_b + key_list[7]) % N)
                    row.append((C_r, C_g, C_b))
                    C = C_r

                else:
                    C = ((key_list[4] + C1) % N) ^ ((key_list[5] + C2) % N) ^ (
                                (key_list[6] + imageMatrix[i][j]) % N) ^ ((C + key_list[7]) % N)
                    row.append(C)

                x = (x + C / 256 + key_list[8] / 256 + key_list[9] / 256) % 1
                y = (x + C / 256 + key_list[8] / 256 + key_list[9] / 256) % 1
                for ki in range(12):
                    key_list[ki] = (key_list[ki] + key_list[12]) % 256
                    key_list[12] = key_list[12] ^ key_list[ki]
            LogisticEncryptionIm.append(row)
            # print(LogisticEncryptionIm)

        im = Image.new("L", (dimensionX, dimensionY))
        if color:
            im = Image.new("RGB", (dimensionX, dimensionY))
        else:
            im = Image.new("L", (dimensionX, dimensionY))  # L is for Black and white pixels

        pix = im.load()
        print(pix)
        for x in range(dimensionX):
            for y in range(dimensionY):
                pix[x, y] = LogisticEncryptionIm[x][y]
        im.save(imageName.split('.')[0] + "_Encrypted.png", "PNG")
        x1 = imageName.split('/')[-1] +"_Encrypted.png"
        x2=x1.split('.')[0]+"_Encrypted.png"


        r = data.id
        st = upload.objects.filter(id=r).update(Encryptedimage=x2)

    LogisticEncryption(path,"abcdef90ghi00jklm")







    return redirect("/encrypt/")









def download(request, id):
    data = upload.objects.get(id=id)
    img = data.Encryptedimage
    print(img)
    print("hiiiiiiiiiiiiiiiiii")
    file_path = f'{settings.MEDIA_ROOT}/{img}'
    print(file_path)
    if os.path.exists(file_path):
        with open(file_path, 'rb') as fh:
            response = HttpResponse(fh.read(), content_type="application/vnd.ms-excel")
            response['Content-Disposition'] = 'inline; filename=' + os.path.basename(file_path)
            messages.info(request, "successfully image Downloaded")
            return response
    raise Http404


# pil_im = Image.open(image + ext, 'r')
# imshow(np.asarray(pil_im), cmap='gray')








def generatekey(request, id):
    da = upload.objects.get(id=id)

    return render(request, 'Admin/generate/keygenerate.html', {'da': da})


def email(request):
    data= upload.objects.all()


    # img = data.Encryptedimage
    # file_path = f'{settings.MEDIA_ROOT}/{img}/'


    return render(request, 'Admin/generate/email.html', {'data':data})

#
# def emailgenerate(request):
#     data = upload.objects.get(id=id)
#
#
#     return render(request, 'Admin/generate/emailgenerate.html',{'data':data})



# # #
# from django.shortcuts import render
# from io import BytesIO
# from django.http import HttpResponse
# from django.template.loader import get_template
# from django.views import View
# from xhtml2pdf import pisa
#
#
# #
# def render_to_pdf(template_src, context_dict={}):
#     template = get_template(template_src)
#     html = template.render(context_dict)
#     result = BytesIO()
#     pdf = pisa.pisaDocument(BytesIO(html.encode("ISO-8859-1")), result)
#     if not pdf.err:
#         return HttpResponse(result.getvalue(), content_type='application/pdf')
#     return None
#
#
# data = {
#     "company": "SURYA INFOMATICS",
#     "address": "Ashok pillar",
#     "city": "chennai",
#     "state": "Tamilnadu",
#     "zipcode": "600034",
#
#     "phone": "0123456789",
#     "email": "Suryainfomatics@gmail.com",
#     "website": "Suryainfomatics.com",
# }
#
#
# # #
# #
# class ViewPDF(View):
#     def get(self,request, *args, **kwargs):
#         pdf = render_to_pdf('Admin/generate/emailgenerate.html',data)
#         return HttpResponse(pdf, content_type='application/pdf')
#
#
# #
# class DownloadPDF(View):
#     def get(self, request, *args, **kwargs):
#         pdf = render_to_pdf('Admin/generate/emailgenerate.html', data)
#
#         response = HttpResponse(pdf, content_type='application/pdf')
#         filename = "Invoice_%s.pdf" % ("12341231")
#         content = "attachment; filename='%s'" % (filename)
#         response['Content-Disposition'] = content
#         return response
#
# #
# def index(request):
# 	context = {}
# 	return render(request, 'app/index.html', context)


# def makepdf(request,id):
#     import os
#     from PIL import Image
#     data = upload.objects.get(id=id)
#     r=data.id
#     img=data.Encryptedimage
#     file_path = f'{settings.MEDIA_ROOT}/{img}'
#     out_dir = "pdf/"
#
#
#     source_dir =  "media/output/"
#
#     for i in os.listdir(out_dir):
#         a=str(i).split(".")[0]
#         b=str(img).split(".")[0]
#         if a == b :
#             x=upload.objects.filter(id=r).update(pdfimage=i)
    # for file in os.listdir(file_path):
    #     if file.split('.')[-1] in ('png', 'jpg', 'jpeg'):
    #         image = Image.open(os.path.join(source_dir, file))
    #         image_converted = image.convert('RGB')
    #         image_converted.save(os.path.join(out_dir, '{0}.pdf'.format(file.split('.')[-2])))
    #         xxx111="out_dir",'{0}.pdf'
    #         st= upload.objects.filter(id=r).update(pdfimage=xxx111)




    # messages.info(request, "PDF CREATED PLEASE SEND THE MAIL")
    # return render(request, 'Admin/generate/email.html' ,{'data':data})

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from django.conf import settings
from django.shortcuts import render
from django.contrib import messages
from PIL import Image
from io import BytesIO
from django.http import HttpResponse, Http404
# from django.conf import settingsfrom reportlab.pdfgen import canvas
from email.message import EmailMessage


def domain_send_mail(request,id):
    import yagmail
    data = upload.objects.get(id=id)
    img=data.Encryptedimage
    path = f'{settings.MEDIA_ROOT}/{img}'
    # subject = 'Your subject here'
    # to = [data.email]
    # from_email = settings.DEFAULT_FROM_EMAIL
    # pdf_file_path = data.pdfimage.path
    # print(pdf_file_path)
    # print(data)

    import imghdr
    message = EmailMessage()
    message['Subject'] = "email_subject"
    message['From'] = "demosample178@gmail.com"
    message['To'] = data.email

    with open(path, 'rb') as file:
        image_data = file.read()

    message.set_content("Email from Python with image attachment")

    message.add_attachment(image_data, maintype='image', subtype=imghdr.what(None, image_data))
    server = smtplib.SMTP("smtp.gmail.com" ,587)

    server.ehlo()

    server.starttls()


    server.login("demosample178@gmail.com", "owiixglnlbxwogcs")

    server.send_message(message)

    server.quit()
    messages.info(request, "SENT SUCESSFULLY")
    return  redirect("/sentmail/")











#
# def domain_send_mail(request,id):
#     datas = upload.objects.get(id=id)
#       # Replace with the actual name of the PDF file field in your model
#     import os
#     m =datas.name
#     subject = 'Your subject here'
#     to = [datas.email]
#     from_email = settings.DEFAULT_FROM_EMAIL
#     file1= datas.pdfimage
#     pdf_file = file1
#     print("the file is",file1)
#     other_dir="pdf"
#     file_path=[]
#     # k = file_path[-1]
#
#     for i in os.listdir(other_dir):
#         print(i)
#         file_path.append(i)
#         print(file_path)
#         for i in file_path:
#             print(i)
#             if i == file1:
#                 print(i)
#
#             # Create an email message
#             #     email_subject = 'Security Alert'
#             #     email_body = 'There has been an attempt to break your accounts security, but dont worry—well take care of it in accordance with our security protocol. Please adhere to the safety precautions. Keep smiling, And stay safe.'
#             #     email_from = 'aakashbsurya@gmail.com'
#             #     email_to = [datas.email]
#             #     email_message = EmailMessage(email_subject, email_body, email_from, email_to)
#             #     with file1.open(mode='rb') as f:
#             #         file_content = f.read()
#             #     # Attach the PDF file to the email message
#             #     email_message.attach(MIMEText(file1, file_content, 'application/pdf'))
#             #
#             #     # Send the email
#             #     email_message.send()
#             #     file = open('newfile.txt')
#             #
#             #     msg.attach(MIMEText(file))
#
#                 import os
#                 import smtplib
#                 from email.mime.text import MIMEText
#                 from email.mime.image import MIMEImage
#                 from email.mime.multipart import MIMEMultipart
#
#                 smtp_ssl_host = 'smtp.gmail.com'  # smtp.mail.yahoo.com
#                 smtp_ssl_port = 465
#                 username = 'USERNAME or EMAIL ADDRESS'
#                 password = 'PASSWORD'
#                 sender = from_email
#                 targets = ['dharaneshwaran.surya@gmail.com', 'selja.surya@gmail.com']
#
#                 msg = MIMEMultipart()
#                 msg['Subject'] = 'I have a picture'
#                 msg['From'] = sender
#                 msg['To'] = 'dharaneshwaran.surya@gmail.com'
#                 print(from_email)
#
#                 txt = MIMEText('I just bought a new camera.')
#                 msg.attach(txt)
#                 filepath = "pdf/output.pdf"
#                 with open(filepath, 'rb') as f:
#                     img = MIMEMultipart(f.read())
#
#                 img.add_header('Content-Disposition',
#                                'attachment',
#                                filename=os.path.join(filepath))
#                 msg.attach(img)
#
#                 server = smtplib.SMTP_SSL(smtp_ssl_host, smtp_ssl_port)
#
#
#                 server.quit()
#                 messages.info(request, " SUCCESSFULLY ")
#
#     return render(request, "Admin/generate/email.html")



def sentmail(request):
    data = upload.objects.all()

    return render(request, "Admin/generate/sent_mail.html", {'data': data})

def makepdf11(request):
    data=upload.objects.all()
    return render(request,"Admin/generate/makepdf11.html", {'data': data})





# def send_email_with_image_attachment():
#     subject = 'Your subject here'
#     to = ['recipient@example.com']
#     from_email = settings.DEFAULT_FROM_EMAIL
#
#     # load the image from a file
#     image_path = '/path/to/your/image.png'
#     with open(image_path, 'rb') as f:
#         image_data = f.read()
#     image_name = 'myimage.png'

    # create the email message

# import os
# from django.core.mail import EmailMultiAlternatives
# from django.template.loader import render_to_string
#
# def send_mail_with_attachment(subject, to, template_name, context, attachment_file_path):
#     message_html = render_to_string(template_name, context)
#     message_txt = strip_tags(message_html)
#     email = EmailMultiAlternatives(subject, message_txt, to=to)
#     email.attach_file(attachment_file_path)
#     email.attach_alternative(message_html, "text/html")
#     email.send()
#
# send_mail_with_attachment(
#     'Subject Line Here',
#     'recipient@example.com',
#     'template.html',
#     {'context': 'var'},
#     os.path.join(os.getcwd(), 'path/to/attachment.pdf')
# )