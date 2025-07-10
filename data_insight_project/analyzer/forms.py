from django import forms
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Layout, Submit, Row, Column, Div, HTML
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django.utils.safestring import mark_safe
from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import UserProfile
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django.contrib import messages

class DataPreprocessingForm(forms.Form):
    # Eksik Veri Doldurma Seçenekleri
    FILL_MISSING_CHOICES = [
        ('none', 'Uygulama'),
        ('mean', 'Ortalama (Mean)'),
        ('median', 'Medyan (Median)'),
        ('mode', 'Mod (Mode)'),
        ('drop', 'Satırı Sil'),
        ('knn', 'KNN ile Doldur'),
    ]
    
    # Normalizasyon Seçenekleri
    SCALING_CHOICES = [
        ('none', 'Uygulama'),
        ('minmax', 'Min-Max'),
        ('standard', 'Standard')
    ]
    
    # Kategorik Veri Dönüşüm Seçenekleri
    ENCODING_CHOICES = [
        ('none', 'Uygulama'),
        ('label', 'Label Encoding'),
        ('onehot', 'One-Hot Encoding')
    ]
    
    # Model Seçenekleri
    MODEL_CHOICES = [
        ('logistic', 'Lojistik Regresyon'),
        ('decision_tree', 'Karar Ağacı'),
        ('random_forest', 'Random Forest'),
        ('svm', 'Destek Vektör Makinesi'),
        ('knn', 'K-En Yakın Komşu'),
        ('naive_bayes', 'Naive Bayes'),
        ('xgboost', 'XGBoost'),
        ('lightgbm', 'LightGBM'),
        ('ridge', 'Ridge (Regresyon)'),
        ('lasso', 'Lasso (Regresyon)'),
    # ... diğerleri ...
]
    # Form Alanları
    fill_missing = forms.ChoiceField(
        choices=FILL_MISSING_CHOICES,
        label=mark_safe('Eksik Veri Doldurma <i class="fas fa-question-circle text-info" data-bs-toggle="tooltip" title="Eksik verileri doldurma yöntemini seçin."></i>'),
        initial='none',
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    
    scaling_method = forms.ChoiceField(
        choices=SCALING_CHOICES,
        label=mark_safe('Normalizasyon <i class="fas fa-question-circle text-info" data-bs-toggle="tooltip" title="Veri ölçeklendirme yöntemini seçin."></i>'),
        initial='none',
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    
    encoding_method = forms.ChoiceField(
        choices=ENCODING_CHOICES,
        label=mark_safe('Kategorik Veri Dönüşümü <i class="fas fa-question-circle text-info" data-bs-toggle="tooltip" title="Kategorik verileri sayısal hale getirme yöntemini seçin."></i>'),
        initial='none',
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    
    remove_constant = forms.BooleanField(
        label=mark_safe('Sabit Sütunları Kaldır <i class="fas fa-question-circle text-info" data-bs-toggle="tooltip" title="Tüm değerleri aynı olan sütunları kaldırır."></i>'),
        required=False,
        initial=True,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    
    remove_high_missing = forms.BooleanField(
        label=mark_safe('Yüksek Eksik Veri İçeren Sütunları Kaldır <i class="fas fa-question-circle text-info" data-bs-toggle="tooltip" title="%80 ve üzeri eksik veri içeren sütunları kaldırır."></i>'),
        required=False,
        initial=True,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    
    train_model = forms.BooleanField(
        label=mark_safe('Model Eğitimi <i class="fas fa-question-circle text-info" data-bs-toggle="tooltip" title="Veriyle otomatik makine öğrenimi modeli eğitmek için işaretleyin."></i>'),
        required=False,
        initial=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    
    target_column = forms.ChoiceField(
        label="Hedef Sütun",
        choices=[],
        required=False,
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    
    model_choice = forms.ChoiceField(
        choices=MODEL_CHOICES,
        label=mark_safe('Model Seçimi <i class="fas fa-question-circle text-info" data-bs-toggle="tooltip" title="Kullanmak istediğiniz makine öğrenimi modelini seçin."></i>'),
        required=False,
        initial='logistic',
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    
    # Model Özel Parametreleri
    dt_max_depth = forms.IntegerField(
        label=mark_safe('Maksimum Derinlik (Karar Ağacı) <i class="fas fa-question-circle text-info" data-bs-toggle="tooltip" title="Karar ağacı için maksimum derinlik."></i>'),
        required=False,
        initial=5,
        min_value=1,
        max_value=20,
        widget=forms.NumberInput(attrs={'class': 'form-control dt-param'})
    )
    
    rf_n_estimators = forms.IntegerField(
        label=mark_safe('Ağaç Sayısı (Random Forest) <i class="fas fa-question-circle text-info" data-bs-toggle="tooltip" title="Random Forest için ağaç (estimators) sayısı."></i>'),
        required=False,
        initial=100,
        min_value=10,
        max_value=1000,
        widget=forms.NumberInput(attrs={'class': 'form-control rf-param'})
    )
    
    reg_alpha = forms.FloatField(
        label=mark_safe('Regularizasyon Parametresi (Lojistik Regresyon) <i class="fas fa-question-circle text-info" data-bs-toggle="tooltip" title="Lojistik regresyon ve benzeri modellerde regularizasyon katsayısı."></i>'),
        required=False,
        initial=1.0,
        min_value=0.0,
        max_value=10.0,
        widget=forms.NumberInput(attrs={'class': 'form-control reg-param'})
    )
    
    knn_n_neighbors = forms.IntegerField(
        label=mark_safe('Komşu Sayısı (KNN) <i class="fas fa-question-circle text-info" data-bs-toggle="tooltip" title="KNN algoritması için komşu sayısı."></i>'),
        required=False,
        initial=5,
        min_value=1,
        max_value=20,
        widget=forms.NumberInput(attrs={'class': 'form-control knn-param'})
    )

    def __init__(self, *args, **kwargs):
        target_column_choices = kwargs.pop('target_column_choices', [])
        super().__init__(*args, **kwargs)
        
        # Hedef sütun seçeneklerini güncelle
        self.fields['target_column'].choices = target_column_choices
        
        # Form düzenini ayarla
        self.helper = FormHelper()
        self.helper.form_method = 'post'
        self.helper.form_class = 'form-horizontal'
        self.helper.label_class = 'col-lg-3'
        self.helper.field_class = 'col-lg-9'
        
        # Form düzeni
        self.helper.layout = Layout(
            HTML('<h5 class="mb-3">Veri Ön İşleme</h5>'),
            Row(
                Column('fill_missing', css_class='form-group'),
                Column('scaling_method', css_class='form-group'),
            ),
            Row(
                Column('encoding_method', css_class='form-group'),
                Column('remove_constant', css_class='form-group'),
            ),
            'remove_high_missing',
            
            HTML('<h5 class="mb-3 mt-4">Model Eğitimi</h5>'),
            'train_model',
            Div(
                'target_column',
                'model_choice',
                'dt_max_depth',
                'rf_n_estimators',
                'reg_alpha',
                'knn_n_neighbors',
                css_class='model-options d-none',
                css_id='model-options'
            ),
            
            Submit('submit', 'Analizi Başlat', css_class='btn btn-primary mt-4')
        )

class CSVUploadForm(forms.Form):
    csv_file = forms.FileField(
        label="CSV Dosyası",
        required=False,  # Örnek veri için False olmalı!
        help_text="Lütfen analiz edilecek CSV dosyasını seçin",
        widget=forms.FileInput(attrs={'class': 'form-control', 'accept': '.csv'})
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_method = 'post'
        self.helper.form_enctype = 'multipart/form-data'
        self.helper.layout = Layout(
            'csv_file',
            Submit('submit', 'Dosyayı Yükle', css_class='btn btn-primary mt-3')
        ) 

GENDER_CHOICES = (
    ('', 'Seçiniz'),
    ('male', 'Erkek'),
    ('female', 'Kadın'),
    ('other', 'Diğer'),
)

class RegisterForm(UserCreationForm):
    first_name = forms.CharField(max_length=30, required=False, label="Ad")
    last_name = forms.CharField(max_length=30, required=False, label="Soyad")
    email = forms.EmailField(required=True, label="E-posta")
    birth_date = forms.DateField(
        required=False,
        label="Doğum Tarihi",
        widget=forms.DateInput(attrs={'type': 'date'})
    )
    gender = forms.ChoiceField(
        choices=GENDER_CHOICES,
        required=False,
        label="Cinsiyet"
    )
    accept_terms = forms.BooleanField(
        required=True,
        label="Kullanım koşullarını ve KVKK metnini okudum, kabul ediyorum"
    )

    class Meta:
        model = User
        fields = ("username", "first_name", "last_name", "email", "birth_date", "gender", "accept_terms", "password1", "password2")

@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        UserProfile.objects.create(user=instance)

class ProfileUpdateForm(forms.ModelForm):
    class Meta:
        model = UserProfile
        fields = ['birth_date', 'gender']

class FullProfileUpdateForm(forms.ModelForm):
    # User modelinden alanlar
    first_name = forms.CharField(max_length=30, required=False, label="Ad")
    last_name = forms.CharField(max_length=30, required=False, label="Soyad")
    email = forms.EmailField(required=True, label="E-posta")
    
    class Meta:
        model = UserProfile
        fields = ['birth_date', 'gender']

    def __init__(self, *args, **kwargs):
        user = kwargs.pop('user', None)
        super().__init__(*args, **kwargs)
        if user:
            self.fields['first_name'].initial = user.first_name
            self.fields['last_name'].initial = user.last_name
            self.fields['email'].initial = user.email

    def save(self, commit=True, user=None):
        profile = super().save(commit=False)
        if user:
            user.first_name = self.cleaned_data['first_name']
            user.last_name = self.cleaned_data['last_name']
            user.email = self.cleaned_data['email']
            if commit:
                user.save()
        if commit:
            profile.save()
        return profile