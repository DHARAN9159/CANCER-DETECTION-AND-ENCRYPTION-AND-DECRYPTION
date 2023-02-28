from . import  views
from django.urls import path

urlpatterns=[
    path('doctorlogi/',views.doctor_logi),
    path('doctorregis/',views.doctor_regis),
    path('doctorhome/',views.doctorhome),
    path('doctorreport/',views.doctorreport),
    path('doctorreport12/',views.doctorreport12),
    path('decrypt/<int:id>/',views.decrypt),


]