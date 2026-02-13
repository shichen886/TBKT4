from django.urls import path
from django.views.generic import TemplateView
from django.conf import settings
import views

urlpatterns = [
    path('', TemplateView.as_view(template_name='index.html'), name='home'),
    path('dashboard/', TemplateView.as_view(template_name='dashboard.html'), name='dashboard'),
    path('recommendation/', views.recommendation_page, name='recommendation'),
    path('analysis/', views.analysis_page, name='analysis'),
    path('learning_path/', views.learning_path_page, name='learning_path'),
    path('assessment/', views.assessment_page, name='assessment'),
    path('upload/', views.upload_page, name='upload'),
    path('prediction/', views.prediction_page, name='prediction'),
    path('model_comparison/', views.model_comparison_page, name='model_comparison'),
    path('login/', views.login_page, name='login'),
    path('register/', views.register_page, name='register'),
    path('name-customization/', TemplateView.as_view(template_name='name-customization.html'), name='name_customization'),
    path('heatmap/', views.heatmap_page, name='heatmap'),
    
    path('api/datasets/', views.api_datasets, name='api_datasets'),
    path('api/dataset/<str:dataset>/', views.api_dataset_info, name='api_dataset_info'),
    path('api/user/<str:dataset>/<int:user_id>/', views.api_user_info, name='api_user_info'),
    path('api/recommend/<str:dataset>/<int:user_id>/', views.api_recommendations, name='api_recommendations'),
    path('api/predict/<str:dataset>/<int:user_id>/', views.api_prediction, name='api_prediction'),
    path('api/skills/<str:dataset>/<int:user_id>/', views.api_skill_stats, name='api_skill_stats'),
    path('api/trend/<str:dataset>/<int:user_id>/', views.api_learning_trend, name='api_learning_trend'),
    path('api/errors/<str:dataset>/<int:user_id>/', views.api_error_analysis, name='api_error_analysis'),
    path('api/path/<str:dataset>/<int:user_id>/', views.api_learning_path, name='api_learning_path'),
    path('api/predict/', views.api_predict, name='api_predict'),
    path('api/ocr/', views.api_ocr, name='api_ocr'),
    path('api/save-question/', views.api_save_question, name='api_save_question'),
    path('api/mappings/<str:dataset>/', views.api_mappings, name='api_mappings'),
    path('api/ids/<str:dataset>/<str:name_type>/', views.api_ids, name='api_ids'),
    path('api/heatmap/<str:dataset>/<int:user_id>/', views.api_heatmap, name='api_heatmap'),
    path('api/save-name/', views.api_save_name, name='api_save_name'),
    path('api/auto-generate-name/', views.api_auto_generate_name, name='api_auto_generate_name'),
    path('api/batch-generate-names/', views.api_batch_generate_names, name='api_batch_generate_names'),
]
