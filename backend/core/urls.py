"""
QuantNova AI — apps/urls.py
All app-level URL routes.
"""

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'analysis/upload', views.ChartUploadViewSet, basename='chart-upload')
router.register(r'analysis/results', views.AnalysisResultViewSet, basename='analysis-result')
router.register(r'journal', views.TradeJournalViewSet, basename='journal')
router.register(r'strategies', views.StrategyDocumentViewSet, basename='strategy')
router.register(r'training/jobs', views.TrainingJobViewSet, basename='training-job')
router.register(r'training/versions', views.ModelVersionViewSet, basename='model-version')

urlpatterns = [
    # Auth
    path('auth/login/', views.LoginView.as_view(), name='login'),
    path('auth/register/', views.RegisterView.as_view(), name='register'),
    path('auth/profile/', views.ProfileView.as_view(), name='profile'),

    # Dashboard
    path('dashboard/', views.DashboardView.as_view(), name='dashboard'),

    # Patterns reference
    path('analysis/patterns/', views.PatternLibraryView.as_view(), name='pattern-library'),

    # Router URLs (ViewSets)
    path('', include(router.urls)),
]

# Full URL map:
#
# POST   /api/auth/register/
# POST   /api/auth/login/
# GET    /api/auth/profile/
# PUT    /api/auth/profile/
#
# GET    /api/dashboard/
#
# POST   /api/analysis/upload/
# GET    /api/analysis/upload/
# GET    /api/analysis/upload/{id}/
# GET    /api/analysis/results/
# GET    /api/analysis/results/{id}/
# GET    /api/analysis/results/stats/
# GET    /api/analysis/patterns/
#
# GET    /api/journal/
# POST   /api/journal/
# GET    /api/journal/{id}/
# PUT    /api/journal/{id}/
# DELETE /api/journal/{id}/
# GET    /api/journal/stats/
# POST   /api/journal/{id}/close/
#
# GET    /api/strategies/
# POST   /api/strategies/
# GET    /api/strategies/{id}/
# GET    /api/strategies/{id}/rules/
# POST   /api/strategies/{id}/toggle-rule/
#
# POST   /api/training/jobs/trigger/
# GET    /api/training/jobs/status/
# GET    /api/training/jobs/
# GET    /api/training/versions/
# GET    /api/training/versions/active/