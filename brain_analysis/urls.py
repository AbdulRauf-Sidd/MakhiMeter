
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from .views import brain_home, brain_output, threshold_image, circle_image


urlpatterns = [
    path("upload/", brain_home, name='brain_home'),
    path("output/", brain_output, name='brain_output'),
    path("circle/", circle_image, name='circle_image'),
    path("threshold/", threshold_image, name='threshold_image'),
]
