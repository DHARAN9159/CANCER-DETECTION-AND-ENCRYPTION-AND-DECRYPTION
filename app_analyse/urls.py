from . import views
from django.urls import path

urlpatterns = [
    path('analys_regis/', views.analyseregister),
    path('analys_logi/', views.analyselogin),
    path('analysehome/', views.analysehome),
    path('anal_patient_details/', views.analysepatientdetails),
    path('sym/<int:id>/', views.sym),
    path('predict/',views.predict),
    path('updateanalyse/<int:id>/',views.updateanalyse),
    path('analysisworld/',views.analysisworld),
    path('ana1/',views.ana1),
    path('ana2/',views.ana2),
    path('alert/',views.alert),
    # path('showpredict/<int:id>/',views.showpredict),
    # path('show_output/', views.show_output, name='show_output'),
    # path('show_predict/', views.show_predict),
    # path('show_output1/<int:id>/', views.show_output1)
    # path('show')
]
