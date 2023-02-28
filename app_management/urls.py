from . import  views
from django.urls import path

urlpatterns=[
    path('admin_logi/',views.adminlogin),
    path('adminhome/', views.admin_home),
    path('dashboard1/',views.dashboard1),
    path('dashboard2/',views.dashboard2),
    path('showimage1/<int:id>/',views.showimage1),
    path('showresult/<int:id>/',views.showresult),
    path('cancer/',views.cancer),
    path('normal/',views.normal),
    path('encrypt/',views.encrypt),
    path('encryptimage/<int:id>/',views.encryptimage),
    path('generatekey/<int:id>/',views.generatekey),
    path('email/',views.email),
    # path('emailgenerate/',views.emailgenerate),
    # path('pdf_view/', views.ViewPDF.as_view(), name="pdf_view"),
    # path('pdf_download/', views.DownloadPDF.as_view(), name="pdf_download"),
    path('sentmail/',views.sentmail),
    path('encry/',views.encry),
    path('download/<int:id>/', views.download),
    # path('makepdf/<int:id>/',views.makepdf),
    # path('makepdf11/',views.makepdf11),
    path('domain_send_mail/<int:id>/',views.domain_send_mail)



]