from . import  views
from django.urls import path

urlpatterns=[
    path('researcher_regis/',views.researcherregister),
    path('researcher_logi/',views.researcherlogin),
    path('researchhome/',views.researchhome),
    path('blur/',views.blur),
    path('blurimage/<int:id>/',views.blurimage),
    path('filter/',views.filter),
    path('filterimage/<int:id>/',views.filterimage),
    path('grayscale/',views.grayscale),
    path('grayscaleimage/<int:id>/',views.grayscaleimage),
    path('resize/',views.resize),
    path('resizeimage/<int:id>/',views.resizeimage),
    path('predictionimage/',views.predictionimage),
    path('delete/<int:id>/',views.delete),
    path('analyse_image/<int:id>/',views.analyse_image),
    # path('show/',views.show),
    path('download/<int:id>/', views.download),
    path('resultimage/',views.resultimage),
    path('show/<int:id>/',views.show),
    path('show1/',views.show1),
    path('blur1/',views.blur1),


]