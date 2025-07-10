from django.urls import path
from . import views
from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404, redirect
from django.contrib.auth.models import User
from django.db import models
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import shutil
import os
from django.conf import settings
from django.conf.urls.static import static

app_name = 'analyzer'

urlpatterns = [
    path('', views.home, name='home'),  # Ana sayfa - örnek analizler
    path('upload/', views.upload_file, name='upload'),
    path('analyze/', views.analyze_data, name='analyze'),
    path('result/', views.show_results, name='result'),
    path('download-report/', views.download_report, name='download_report'),
    path('download-processed-data/', views.download_processed_data, name='download_processed_data'),
    path('register/', views.register, name='register'),
    path('analysis/<int:analysis_id>/', views.analysis_detail, name='analysis_detail'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('profile/edit/', views.profile_edit, name='profile_edit'),
    path('change-password/', views.change_password, name='change_password'),
    path('save-analysis/', views.save_analysis, name='save_analysis'),
    path('analysis/<int:analysis_id>/edit/', views.edit_analysis, name='edit_analysis'),
    path('analysis/<int:analysis_id>/delete/', views.delete_analysis, name='delete_analysis'),
    path('analysis/<int:analysis_id>/favorite/', views.toggle_favorite, name='toggle_favorite'),
    path('my-analyses/', views.my_analyses, name='my_analyses'),
    path('public-analyses/', views.public_analyses, name='public_analyses'),
    path('upload_example/', views.upload_example, name='upload_example'),
    path('faq/', views.faq, name='faq'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT) 