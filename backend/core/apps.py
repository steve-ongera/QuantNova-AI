"""
QuantNova AI — apps/apps.py
Django application configuration.
"""

from django.apps import AppConfig


class AppsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name  = 'apps'
    label = 'apps'
    verbose_name = 'QuantNova AI'

    def ready(self):
        """
        Called when Django starts.
        Import signals here to ensure they are registered.
        """
        try:
            import apps.signals  # noqa: F401
        except ImportError:
            pass