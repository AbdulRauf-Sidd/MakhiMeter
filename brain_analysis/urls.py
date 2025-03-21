
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from .views import brain_home, brain_output


urlpatterns = [
    path("upload/", brain_home, name='brain_home'),
    path("output/", brain_output, name='brain_output'),
]
