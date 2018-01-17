from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^$', views.classifier, name='classifier'),
    url(r'^training/', views.training, name='training'),

]
