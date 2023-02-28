from django.shortcuts import render,redirect
from django.contrib import messages
from. models import *
from collections import Counter
# Create your views here.

def userhome(request):
    messages.info(request, 'Login sucessfully.')
    return render(request ,'user/try.html')


def logi(request):
    if request.method=="POST":
        email=request.POST['email']
        password=request.POST['password']
        try:

            te = user_register.objects.get(email=email, password=password)
            request.session['patient'] = te.email

            messages.info(request, 'Sucessfully login')
            return redirect('/userhome/')

        except:
            messages.info("please try again")
            return redirect('/log/')

    return render(request ,'user/login.html')




def regis(request):
    if request.method=="POST":
        name = request.POST['name']
        email =request.POST['email']
        phone_number = request.POST['phone_number']
        password =request.POST['password']
        confirm_password =request.POST['confirm_password']
        user_register(name=name,email=email,password=password,phone_number=phone_number,
              confirm_password=confirm_password ).save()

        messages.info(request, 'registered sucessfully.')



    return render(request,'user/register.html')




def basicdetails(request):
    if request.method=="POST":
        reg = user_register.objects.get(email=request.session['patient'])
        basic = basic_details()
        basic.patientname= reg.name
        basic.email=reg.email
        basic.phonenumber=reg.phone_number
        basic.birthdate =request.POST.get('birthdate')
        basic.gender=request.POST.get('gender')
        basic.age=request.POST.get('age')
        basic.marriedstatus=request.POST.get('marriedstatus')
        basic.spousename=request.POST.get('spousename')
        basic.address=request.POST.get('address')
        basic.city=request.POST.get('city')
        basic.state=request.POST.get('state')
        basic.postal=request.POST.get('postal')
        basic.country=request.POST.get('country')
        basic.fathername=request.POST.get('fathername')
        basic.mothername=request.POST.get('mothername')
        basic.guardianname=request.POST.get('guardianname')
        basic.parentsnumber=request.POST.get('parentsnumber')
        basic.guardiannumber=request.POST.get('guardiannumber')
        basic.guardianrelationship=request.POST.get('guardianrelationship')

        basic.save()
        messages.info(request, 'Submitted sucessfully.')

        def __str__(self):
            return "Hello " + self.patientname


    return render(request,'user/Basicdetails/Basic_details.html')

def patientmedicaldetails(request):
    if request.method == "POST":
        firstcheckup=request.POST['firstcheckup']
        othertest=request.POST['othertest']
        digestive=request.POST['digestive']
        kord=request.POST['kord']
        symptoms=request.POST['symptoms']
        algeric=request.POST['algeric']
        multiple=request.POST['multiple']
        genetic=request.POST['genetic']
        bloodgroup=request.POST['bloodgroup']
        tobacco=request.POST['tobacco']
        alcohol=request.POST['alcohol']
        surgical=request.POST['surgical']
        surgicalmessage=request.POST['surgicalmessage']


        medicaldetails(firstcheckup=firstcheckup,othertest=othertest,digestive=digestive,kord=kord,symptoms=symptoms,algeric=algeric,multiple=multiple,
                       genetic=genetic,bloodgroup=bloodgroup,tobacco=tobacco,alcohol=alcohol,surgical=surgical,surgicalmessage=surgicalmessage).save()

        if firstcheckup and othertest =='no':
            print("if")
            return redirect('/uploaddetails/')
        else:
            print("else")
            return redirect('/alert/')
        messages.info(request, 'Submitted sucessfully.')
    return render(request,'user/addmedical/addmedical.html')


def uploaddetails(request):
    if request.method=="POST":
        pat = user_register.objects.get(email=request.session['patient'])
        print(pat.name)
        img = upload()
        img.name=pat.name
        img.email=pat.email
        img.image = request.FILES['image']
        img.save()
        messages.info(request, 'Upload sucessfully.')
    return  render(request,'user/upload/Upload.html')



def databasefile(request):
    data = medicaldetails.objects.all()


    return render(request, 'user/Databasefile/datab.html', {'data': data})




