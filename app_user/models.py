from django.db import models



class user_register(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
    phone_number = models.PositiveIntegerField()
    password = models.CharField(max_length=10)
    confirm_password = models.CharField(max_length=10)


class basic_details(models.Model):
    patientname = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
    phonenumber = models.PositiveIntegerField(null=True)
    birthdate =models.CharField(max_length=100)
    gender =models.CharField(max_length=20)
    age = models.PositiveIntegerField(null=True)
    marriedstatus=models.CharField(max_length=100,null=True)
    spousename=models.CharField(max_length=100,null=True)
    address =models.CharField(max_length=100)
    city = models.CharField(max_length=100)
    state =models.CharField(max_length=100)
    postal=models.PositiveIntegerField(null=True)
    country=models.CharField(max_length=100)
    fathername=models.CharField(max_length=100)
    mothername=models.CharField(max_length=100)
    guardianname=models.CharField(max_length=100)
    parentsnumber=models.PositiveIntegerField(null=True)
    guardiannumber=models.PositiveIntegerField(null=True)
    guardianrelationship=models.CharField(max_length=250 ,null=True)
    patdieases=models.CharField(max_length=200,null=True)
    pat_sym=models.CharField(max_length=200,null=True)
    showData=models.BooleanField(default=False)



class medicaldetails(models.Model):
    firstcheckup=models.CharField(max_length=10)
    othertest=models.CharField(max_length=10)
    digestive=models.CharField(max_length=10,null=True)
    kord=models.CharField(max_length=10,null=True)
    symptoms=models.CharField(max_length=200)
    algeric=models.CharField(max_length=10,null=True)
    multiple=models.CharField(max_length=300)
    genetic=models.CharField(max_length=100)
    bloodgroup=models.CharField(max_length=10)
    tobacco=models.CharField(max_length=10)
    alcohol=models.CharField(max_length=10)
    surgical=models.CharField(max_length=10)
    surgicalmessage=models.CharField(max_length=200)
    disease = models.CharField(max_length=200,null=True)

# class join1(models.Model):
#     id = models.ForeignKey(basic_details, on_delete=models.CASCADE)
#     id1 = models.ForeignKey(medicaldetails, on_delete=models.CASCADE)

class upload(models.Model):
    name = models.CharField(max_length=200, null=True)
    email=models.EmailField(null=True,unique=True)
    image = models.FileField(upload_to='image/', null=True, blank=True)
    result=models.CharField(max_length=100,null=True)
    key=models.IntegerField(null=True)
    Encryptedimage = models.ImageField(null=True, blank=True)
    pdfimage = models.FileField(null=True,blank=True)
    Decryptedimage=models.ImageField(null=True,blank=True)





