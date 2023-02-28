from django.db import models

# Create your models here.


class analyser_register(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
    phone_number = models.PositiveIntegerField()
    password = models.CharField(max_length=10)
    confirm_password = models.CharField(max_length=10)

