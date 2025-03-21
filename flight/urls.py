
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from .views import flight_upload, flight_output


urlpatterns = [
    path("upload/", flight_upload, name='flight_home'),
    path("output/", flight_output, name='test'),
]
