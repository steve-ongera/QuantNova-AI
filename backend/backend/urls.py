"""
QuantNova AI — config/urls.py
Main project URL configuration.
"""

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    # Django Admin
    path('admin/', admin.site.urls),

    # All API routes live under /api/
    path('api/', include('core.urls')),

    # JWT token refresh
    path('api/auth/token/refresh/', include('rest_framework_simplejwt.urls')),
]

# Serve media files in development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)