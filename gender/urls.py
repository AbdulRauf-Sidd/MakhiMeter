
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from .views import gender_output, gender_upload


urlpatterns = [
    path("upload/", gender_upload, name='gender_upload'),
    path("output/", gender_output, name='gender_output'),
]
