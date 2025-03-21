from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('upload/', views.wing_upload, name='wing_upload'),
    # path('upload/', views.wing_upload, name='wing_upload'),
    path('download/<int:wing_id>/', views.download_results, name='download_results'),
    path('test/', views.test, name='testing'),
    # path('home/', views.home, name='home'),
    # path('login/', views.login, name='login'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
