from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    
    path('upload/', views.image_upload_view, name='upload_image'),
    path('home/', views.home, name='home'),
    path('login/', views.login, name='login'),
    path('input/', views.wing_input, name='wing_input'),
    path('result/', views.result, name='results'),
    path('test/', views.test, name='test'),
    # path('dilate/', views.upload_image, name='upload_image2'),
    path('process-image/', views.process_image, name='process_image'),
    # path('process-static-image/', views.process_image_from_static, name='process_static_image'),
    path('image/', views.show_image_processor, name='image_processor'),
    path('save-static-image/', views.save_image_from_static, name='saver'),
    # path('image/<path:image_url>/', views.display_image, name='display_image'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
