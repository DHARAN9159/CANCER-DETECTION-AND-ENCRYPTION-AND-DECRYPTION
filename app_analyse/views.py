from django.shortcuts import render, redirect
from .models import *
from app_user.models import *
from django.contrib import messages
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron

# Create your views here.
def analyseregister(request):
    if request.method == "POST":
        name = request.POST['name']
        email = request.POST['email']
        phone_number = request.POST['phone_number']
        password = request.POST['password']
        confirm_password = request.POST['confirm_password']
        analyser_register(name=name, email=email, password=password, phone_number=phone_number,
                          confirm_password=confirm_password).save()
        messages.info(request, 'registered sucessfully.')

    return render(request, 'analyse/analyse_regis.html')


def analyselogin(request):
    if request.method == "POST":
        email = request.POST['email']
        password = request.POST['password']
        try:

            te = analyser_register.objects.get(email=email, password=password)

            messages.info(request, 'login Sucessfully ')
            return redirect('/analysehome/')

        except:
            return redirect('/analyselogin/')



    return render(request, 'analyse/analyse_login.html')


def analysehome(request):
    messages.info(request, 'login Sucessfully ')
    return render(request, 'analyse/base_home.html')


def analysepatientdetails(request):
    data = medicaldetails.objects.all()
    return render(request, 'analyse/Analyse_patient_data.html', {'data': data})


def algorithm(datas,r):
    data = pd.read_csv('dataset/CANCERDATA.csv')
    data_x = data.iloc[:, :-1]
    data_y = data.iloc[:, -1]
    string_datas = [i for i in data_x.columns if data_x.dtypes[i] == np.object_]

    LabelEncoders = []
    for i in string_datas:
        newLabelEncoder = LabelEncoder()
        data_x[i] = newLabelEncoder.fit_transform(data_x[i])
        LabelEncoders.append(newLabelEncoder)
    ylabel_encoder = None
    if type(data_y.iloc[1]) == str:
        ylabel_encoder = LabelEncoder()
        data_y = ylabel_encoder.fit_transform(data_y)

    model = LogisticRegression()
    model.fit(data_x, data_y)
    value = {data_x.columns[i]: datas[i] for i in range(len(datas))}
    l = 0
    for i in string_datas:
        z = LabelEncoders[l]
        value[i] = z.transform([value[i]])[0]
        l += 1
    value = [i for i in value.values()]
    predicted = model.predict([value])
    if ylabel_encoder:
        predicted = ylabel_encoder.inverse_transform(predicted)

    return predicted[0]

def sym(request,id):
    print('hi')
    data = medicaldetails.objects.get(id=id)
    # symptoms = data.symptoms
    r = data.id
    k = []
    p=data.symptoms
    q=data.tobacco
    s=data.alcohol
    k.append(p)
    k.append(q)
    k.append(s)
    m = algorithm(k,r)

    st = medicaldetails.objects.filter(id=r).update(disease=m)
    print(m)
    return redirect("/analysehome/")





def alert(request):
    return  render(request,'user/alert.html')






        # a = df.plot.scatter(x='Sugar', y='Molasses(Kg)', c='green')
        # # plt.show()
        # b = df.plot.scatter(x='water level', y='Molasses(Kg)', c='red')
        # # plt.show()
        # c = df.plot.scatter(x='Urea', y='Molasses(Kg)', c='black')
        # # plt.show()
        # d = df.plot.scatter(x='Chemical(MG)', y='Molasses(Kg)')
        # # plt.show()
        # e = df1['microbes'].value_counts(normalize=True).plot.bar()
        # g = np.random.choice([a, b, c, d])
        #
        # print(g)
        # plt.show()
        # return redirect('/view_yeast_cal/')
        #
        #






def predict(request):
    data = basic_details.objects.all()
    # othershow= medicaldetails.objects.all()
    return render(request,'analyse/predic.html',{'data':data})

def updateanalyse(request,id):
    data = basic_details.objects.get(id=id)
    data1 = medicaldetails.objects.get(id=id)
    r = data.id
    k = data1.disease
    m = data1.symptoms
    print(m)
    print('hi')
    print(data.email)
    print(data1.disease)
    st = basic_details.objects.filter(id=r).update(patdieases=k)
    st1 = basic_details.objects.filter(id=r).update(pat_sym=m)
    print(k)
    return redirect('/predict/')




#
def ana1(request):

    all_cancer = pd.read_csv("dataset/Global cancer incidence both sexes.csv")
    # print(all_cancer.head())
    # print("------------------")
    female = pd.read_csv("dataset/Global cancer incidence in women.csv")
    # print(female.head())
    # print("-----------------")
    male = pd.read_csv('dataset/Global cancer incidence in men.csv')
    # print(male.head())

    female = female.drop(["Unnamed: 0", "Rank"], 1)
    male = male.drop(["Unnamed: 0", "Rank"], 1)

    x = male['New_cases_in_2020']
    y = female["New_cases_in_2020"]
    # plt.figure()
    # plt.plot(x)
    # plt.plot(y)
    # plt.xlabel('male new case 2020')
    # plt.ylabel('female new case 2020')
    # plt.title('all cancers')
    # plt.show()

    # u = male['New_cases_in_2020']
    # v = female["New_cases_in_2020"]
    # plt.figure()
    # plt.hist(u)
    # plt.hist(v)
    # plt.xlabel('male new case 2020')
    # plt.ylabel('female new case 2020')
    # plt.title('all cancers')
    # plt.show()

    plt.pie(x, labels=male, autopct="%0.2f%%")
    plt.title('male cancer')
    plt.show()

    # plt.pie(y, labels=female, autopct="%0.2f%%")
    # plt.title('female cancer')
    # plt.show()

    return render( request,"analyse/Analysis_world.html")
def ana2(request):

    # all_cancer = pd.read_csv("dataset/Global cancer incidence both sexes.csv")
    # # print(all_cancer.head())
    # print("------------------")
    female = pd.read_csv("dataset/Global cancer incidence in women.csv")
    # print(female.head())
    # print("-----------------")
    male = pd.read_csv('dataset/Global cancer incidence in men.csv')
    # print(male.head())

    female = female.drop(["Unnamed: 0", "Rank"], 1)
    male = male.drop(["Unnamed: 0", "Rank"], 1)

    x = male['New_cases_in_2020']
    y = female["New_cases_in_2020"]
    plt.pie(y, labels=female, autopct="%0.2f%%")
    plt.title('female cancer')
    plt.show()

    return render(request, "analyse/Analysis_world.html")

def analysisworld(request):


    return render (request,"analyse/Analysis_world.html")




# def get_medical_details(request):
#     data = basic_details.objects.filter(analyse=False)
#     return render(request)


# def show(request,id):
#     data=basic_details.objects.get(id=id)
#     return render(request,'analyse/predicted.html',{'data':data})

# def show_output(request):
#     data = basic_details.objects.filter(showData=True)
#     return render (request,'analyse/show_output.html',{'data':data})
#
#
# def show_output1(request,id):
#     data = basic_details.objects.get(id=id)
#     data.showData = True
#     data.save()
#     return redirect('/show_predict/')
#
#
# def show_predict(request):
#     data = basic_details.objects.filter(showData=True)
#     return render(request,'analyse/try.html',{'data':data})
