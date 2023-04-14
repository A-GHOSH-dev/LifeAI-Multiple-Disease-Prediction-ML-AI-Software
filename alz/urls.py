from django.urls import path
from . import views
from django.contrib import admin
from django.urls import path, include
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('heart', views.heart, name="heart"),
    path('diabetes', views.diabetes, name="diabetes"),
    path('breast', views.breast, name="breast"),
    path('', views.home, name="home"),
    
    #alzheimers
    
    path('alhome',views.alhome, name="alhome"),
    path('predict/',views.predict),
    path('predict/predict',views.predict),
    path('predict/res',views.res),
    path('predict/alhome',views.alhome),
    
    #end
    
    #parkinson
    
    path('indexp', views.indexp, name="indexp"),
    path('ppredict/', views.ppredict, name="ppredict"),
    
    #end
    
    #lung
    path('lhome', views.lhome, name="lhome"),
    path('lresult/', views.lresult, name="lresult"),
    #end
]

if settings.DEBUG:
        urlpatterns += static(settings.MEDIA_URL,
                              document_root=settings.MEDIA_ROOT)

