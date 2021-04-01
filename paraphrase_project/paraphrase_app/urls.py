from django.urls import path, include
from . import views
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    # path('', views.home_page, name="home_page"),
    path('', views.HomePage.as_view(), name="class_home_page"),
]
