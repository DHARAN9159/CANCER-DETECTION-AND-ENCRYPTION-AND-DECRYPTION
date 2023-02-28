from . import  views
from django.urls import path

#
urlpatterns=[
    path('log/',views.logi),
    path('regis/',views.regis),
    path('userhome/',views.userhome),
    path('basicdetails/',views.basicdetails),
    path('patientmedicaldetails/',views.patientmedicaldetails),
    path('uploaddetails/',views.uploaddetails),
    path('databasefile/',views.databasefile),





]