"""
URL configuration for makhi_meter project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from wing_segmentation.views import home, login_view, register, logout_view
# from wing_segmentation.views import image_color


urlpatterns = [
    path("admin/", admin.site.urls),
    path("wing/", include('wing_segmentation.urls')),
    path("brain/", include('brain_analysis.urls')),
    path("flight/", include('flight.urls')),
    path("gender/", include('gender.urls')),
    path("", home, name='home'),
    path("logout_user/", logout_view, name='logout'),
    path("login_user", login_view, name='login'),
    path("register", register, name='register'),
    # path("image_colorization", image_color, name='color')
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
