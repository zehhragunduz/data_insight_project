from django.shortcuts import render, redirect
from django.conf import settings
from django.http import HttpResponse, FileResponse, JsonResponse
from django.contrib import messages
from .forms import CSVUploadForm, DataPreprocessingForm
from .utils import (
    load_and_validate_csv,
    analyze_data as perform_data_analysis,
    generate_plots,
    preprocess_data,
    train_model
)
from django.contrib.auth.forms import PasswordChangeForm
from django.contrib.auth import update_session_auth_hash
import pandas as pd
from xhtml2pdf import pisa
from django.template.loader import get_template
import json
from datetime import datetime
import numpy as np
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_POST
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
# from .models import Project  # Proje kaydetme/yükleme için eklenmişti, kaldırıldı
from django.core.serializers.json import DjangoJSONEncoder
import os
from django.shortcuts import render, redirect
from .forms import RegisterForm
from django.contrib.auth.models import User
from .forms import ProfileUpdateForm
from .models import Analysis
import traceback
from scipy import stats
from .utils import guess_target_column
import numpy as np
from django.shortcuts import get_object_or_404
from django.views.decorators.csrf import csrf_exempt
import sys
import math
import shutil
from django import forms
from django.contrib.auth.models import User
from .models import UserProfile
from datetime import date
from django.contrib.auth.views import LoginView
from django.shortcuts import redirect
import random
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.db.models import Count





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
        # Avatar dosyasını işle
        if self.cleaned_data.get('avatar'):
            profile.avatar = self.cleaned_data['avatar']
            profile.avatar_choice = ''  # Kendi fotoğrafı seçilirse avatar_choice sıfırlansın
        elif self.cleaned_data.get('avatar_choice'):
            profile.avatar_choice = self.cleaned_data['avatar_choice']
            profile.avatar = None  # Avatar seçilirse yüklenen dosya sıfırlansın
        if commit:
            profile.save()
        return profile

def home(request):
    analyses_qs = Analysis.objects.filter(is_public=True).order_by('-created_at')
    paginator = Paginator(analyses_qs, 9)  # 9 analiz/sayfa
    page_number = request.GET.get('page')
    try:
        page_obj = paginator.get_page(page_number)
    except PageNotAnInteger:
        page_obj = paginator.page(1)
    except EmptyPage:
        page_obj = paginator.page(paginator.num_pages)
    return render(request, 'analyzer/home.html', {
        'shared_analyses': page_obj.object_list,
        'page_obj': page_obj,
        'paginator': paginator,
    })
def analysis_detail(request, analysis_id):
    analysis = get_object_or_404(Analysis, id=analysis_id)
    if not analysis.is_public and (not request.user.is_authenticated or analysis.owner != request.user):
        return redirect('analyzer:dashboard')
    # Özet ve metrikler için kolay erişim
    summary = analysis.details.get('summary', {})
    metrics = analysis.details.get('model_metrics', {})
    plots = analysis.details.get('plots', {})  # {'confusion_matrix': 'base64...', ...}
    return render(request, "analyzer/analysis_detail.html", {
        "analysis": analysis,
        "summary": summary,
        "metrics": metrics,
        "plots": plots,
    })
@login_required
def change_password(request):
    if request.method == 'POST':
        form = PasswordChangeForm(user=request.user, data=request.POST)
        if form.is_valid():
            user = form.save()
            update_session_auth_hash(request, user)  # Oturumun açık kalmasını sağlar
            messages.success(request, 'Şifreniz başarıyla değiştirildi.')
            return redirect('analyzer:dashboard')
    else:
        form = PasswordChangeForm(user=request.user)
    return render(request, 'analyzer/change_password.html', {'form': form})
@login_required
def profile_edit(request):
    if not hasattr(request.user, 'userprofile'):
        from .models import UserProfile
        UserProfile.objects.create(user=request.user)
    profile = request.user.userprofile
    form = FullProfileUpdateForm(request.POST or None, request.FILES or None, instance=profile, user=request.user)
    password_form = PasswordChangeForm(user=request.user, data=request.POST if request.POST.get('change_password') else None)
    if request.method == 'POST':
        if request.POST.get('change_password'):
            if password_form.is_valid():
                user = password_form.save()
                update_session_auth_hash(request, user)
                messages.success(request, 'Şifreniz başarıyla değiştirildi.')
            else:
                messages.error(request, 'Şifre değiştirme formunda hata var.')
        else:
            if form.is_valid():
                form.save(user=request.user)
                messages.success(request, "Profiliniz başarıyla güncellendi.")
            else:
                messages.error(request, 'Profil formunda hata var.')
    return render(request, 'analyzer/profile_edit.html', {
        'form': form,
        'password_form': password_form,
    })
@login_required
def dashboard(request):
    user_profile = request.user.userprofile
    today = date.today()
    is_birthday = False
    show_birthday_celebration = False

    if user_profile.birth_date:
        if user_profile.birth_date.day == today.day and user_profile.birth_date.month == today.month:
            is_birthday = True
            # Sadece ilk girişte göster
            if not request.session.get('birthday_celebrated', False):
                show_birthday_celebration = True
                request.session['birthday_celebrated'] = True
    else:
        # Doğum günü yoksa kutlama gösterme
        request.session['birthday_celebrated'] = False

    user_analyses = Analysis.objects.filter(owner=request.user).order_by('-created_at')[:5]
    return render(request, "analyzer/dashboard.html", {
        "user_analyses": user_analyses,
        "user": request.user,
        "is_birthday": is_birthday,
        "show_birthday_celebration": show_birthday_celebration,
    })

def register(request):
    if request.method == "POST":
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            user.userprofile.birth_date = form.cleaned_data.get('birth_date')
            user.userprofile.gender = form.cleaned_data.get('gender')
            user.userprofile.accept_terms = form.cleaned_data.get('accept_terms')
            user.userprofile.save()
            messages.success(request, "Kayıt başarılı! Giriş yapabilirsiniz.")
            return redirect("login")
    else:
        form = RegisterForm()
    return render(request, "registration/register.html", {"form": form})

def convert_numpy_types(obj):
    """
    NumPy int64 ve float64 tiplerini standart Python int/float tiplerine dönüştürür.
    Ayrıca JSON'a dönüştürülemeyen diğer veri tiplerini de temizler.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(elem) for elem in obj]
    elif isinstance(obj, (pd.DataFrame, pd.Series)):
        return obj.to_dict() if hasattr(obj, 'to_dict') else str(obj)
    elif hasattr(obj, 'dtype'):  # NumPy array benzeri objeler
        return obj.tolist() if hasattr(obj, 'tolist') else str(obj)
    elif obj is None:
        return None
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    else:
        # Diğer tüm objeleri string'e çevir
        try:
            return str(obj)
        except:
            return None
        
def clean_json(obj):
    import math
    if isinstance(obj, dict):
        return {str(k): clean_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_json(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return str(obj)
        return obj
    elif obj is None:
        return ""
    return obj

@login_required(login_url='login')
def upload_file(request):
    """
    CSV dosyası yükleme sayfası
    """
    if request.method == 'POST':
        example_dataset = request.POST.get('example_dataset')
        print("POST:", request.POST)
        if example_dataset:
            print("example_dataset:", example_dataset)
            # Örnek dosya seçilmişse, media klasöründen kopyala ve session'a kaydet
            example_path = os.path.join(settings.MEDIA_ROOT, example_dataset)
            print("DEBUG - example_path exists:", os.path.exists(example_path))
            if not os.path.exists(example_path):
                messages.error(request, 'Seçilen örnek dosya bulunamadı.')
                return redirect('analyzer:upload')
            # Kullanıcıya özel bir kopya oluştur
            user_file_path = os.path.join(settings.MEDIA_ROOT, f"{request.user.username}_{example_dataset}")
            shutil.copy(example_path, user_file_path)
            request.session['csv_file_path'] = user_file_path
            return redirect('analyzer:analyze')
        form = CSVUploadForm(request.POST, request.FILES)
        if form.is_valid():
            csv_file = request.FILES.get('csv_file')
            if not csv_file:
                messages.error(request, 'Dosya seçilmedi veya yüklenemedi. Lütfen tekrar deneyin.')
                return redirect('analyzer:upload')
            file_path = os.path.join(settings.MEDIA_ROOT, csv_file.name)
            with open(file_path, 'wb+') as destination:
                for chunk in csv_file.chunks():
                    destination.write(chunk)
            try:
                df, warnings = load_and_validate_csv(file_path)
                for warning in warnings:
                    messages.warning(request, warning)
                request.session['csv_file_path'] = file_path
                return redirect('analyzer:analyze')
            except Exception as e:
                messages.error(request, str(e))
                if os.path.exists(file_path):
                    os.remove(file_path)
                return redirect('analyzer:upload')
    else:
        form = CSVUploadForm()
    return render(request, 'analyzer/upload.html', {'form': form})


def remove_outliers(df, columns, method='zscore', threshold=3):
      if method == 'zscore':
          for col in columns:
              z_scores = np.abs(stats.zscore(df[col].dropna()))
              df = df[(z_scores < threshold) | (df[col].isnull())]
      # IQR yöntemi de eklenebilir
      return df


@login_required(login_url='login')
def analyze_data(request):
    """
    Veri analizi ve ön işleme sayfası
    """
    applied_operations_summary = {}
    model_error = None
    if request.method == 'POST':
        if 'csv_file_path' not in request.session:
            messages.error(request, 'Lütfen önce bir CSV dosyası yükleyin.')
            return redirect('analyzer:upload')
        
        file_path = request.session['csv_file_path']
        if not os.path.exists(file_path):
            messages.error(request, 'Yüklenen dosya bulunamadı.')
            return redirect('analyzer:upload')
        
        try:
            df, _ = load_and_validate_csv(file_path)
            all_columns = df.columns.tolist()
            
            target_column_choices = [(col, col) for col in all_columns]
            preprocessing_form = DataPreprocessingForm(
                request.POST or None,
                target_column_choices=target_column_choices
            )
            if preprocessing_form.is_valid():
                options = {
                    'fill_missing': preprocessing_form.cleaned_data['fill_missing'],
                    'scaling_method': preprocessing_form.cleaned_data['scaling_method'],
                    'encoding_method': preprocessing_form.cleaned_data['encoding_method'],
                    'remove_constant': preprocessing_form.cleaned_data['remove_constant'],
                    'remove_high_missing': preprocessing_form.cleaned_data['remove_high_missing'],
                    'train_model': preprocessing_form.cleaned_data['train_model'],
                    'target_column': preprocessing_form.cleaned_data['target_column'],
                    'model_choice': preprocessing_form.cleaned_data['model_choice'],
                    'dt_max_depth': preprocessing_form.cleaned_data['dt_max_depth'],
                    'rf_n_estimators': preprocessing_form.cleaned_data['rf_n_estimators'],
                    'reg_alpha': preprocessing_form.cleaned_data['reg_alpha'],
                    'knn_n_neighbors': preprocessing_form.cleaned_data['knn_n_neighbors']
                }
                # Hedef sütun seçilmemişse otomatik belirle
                if not options['target_column']:
                    options['target_column'] = guess_target_column(df)
                    if not options['target_column']:
                        messages.error(request, "Hedef sütun otomatik olarak belirlenemedi. Lütfen hedef sütunu seçin.")
                        return redirect('analyzer:analyze')
                    else:
                        messages.info(request, f"Hedef sütun otomatik olarak '{options['target_column']}' olarak seçildi.")
                
                # Veri ön işleme
                df_processed, preprocessing_info = preprocess_data(df, options)
                analysis = perform_data_analysis(df_processed)
                plots = generate_plots(df_processed)
                
                is_public = False  # Varsayılan olarak private
                if df_processed.isnull().any().any():
                    raise Exception("Model eğitiminden önce eksik veri (NaN) kalmamalı!")

                # Model eğitimi
                model_metrics = None
                model_error = None
                if options['train_model']:
                    try:
                        model_metrics, model_error,plots = train_model(
                            df_processed, 
                            options['target_column'], 
                            options['model_choice'],
                            options
                        )
                        
                    except Exception as e:
                        messages.warning(request, f'Model eğitimi sırasında bir hata oluştu: {str(e)}')
                        model_metrics = None
                        model_error = str(e)

                
                # Session verilerini hazırla
               
                applied_operations_summary = {
                    'missing_data_filled': {
                        'applied': options['fill_missing'] != 'none',
                        'method': options['fill_missing'] if options['fill_missing'] != 'none' else None,
                        'filled_cells_count': preprocessing_info.get('filled_missing_cells', 0)
                    },
                    'categorical_encoding': {
                        'applied': options['encoding_method'] != 'none',
                        'method': options['encoding_method'] if options['encoding_method'] != 'none' else None,
                        'encoded_columns': preprocessing_info.get('encoded_columns', [])
                    },
                    'normalization': {
                        'applied': options['scaling_method'] != 'none',
                        'method': options['scaling_method'] if options['scaling_method'] != 'none' else None
                    },
                    'outlier_handling': {
                        'applied': False,
                        'method': None
                    },
                    'model_training': {
                        'applied': options['train_model'],
                        'model_type': options['model_choice'] if options['train_model'] else None
                    },
                    'removed_constant_columns': {
                        'applied': options['remove_constant'],
                        'columns': preprocessing_info.get('removed_constant_columns', [])
                    },
                    'removed_high_missing_columns': {
                        'applied': options['remove_high_missing'],
                        'columns': preprocessing_info.get('removed_high_missing_columns', [])
                    }
                }
                session_analysis = convert_numpy_types(analysis)
                session_preprocessing_info = convert_numpy_types(preprocessing_info)
                session_model_metrics = convert_numpy_types(model_metrics)
                session_applied_operations_summary = convert_numpy_types(applied_operations_summary)
                session_plots = convert_numpy_types(plots)
                # Session'a kaydet
                request.session['analysis_results'] = {
                    'analysis': session_analysis,
                    'plots': session_plots,
                    'preprocessing_info': session_preprocessing_info,
                    'model_metrics': session_model_metrics,
                    'model_error': model_error,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'df_processed_data': df_processed.to_json(orient='split'),
                    'applied_operations_summary': session_applied_operations_summary,
                    'file_name': os.path.basename(file_path),
                    'row_count': len(df_processed),
                    'column_count': len(df_processed.columns),
                    'target_column': str(options['target_column']),
                    'model_type': str(options['model_choice']),
                    'short_summary': f"Veri setinde {len(df_processed)} satır ve {len(df_processed.columns)} sütun var.",
                }
                details = {
                    'target_column': str(options['target_column']),
                    'model_type': str(options['model_choice']),
                    'short_summary': f"Veri setinde {len(df_processed)} satır ve {len(df_processed.columns)} sütun var.",
                }

# Sadece bu temel bilgileri kaydet, diğerlerini session'da tut

                # Debug için ekle:
                import json
                try:
                    json.dumps(details)
                    print("Details JSON'a dönüştürülebilir")
                except Exception as e:
                    print(f"JSON hatası: {e}")
                    # Hangi anahtarın sorun yarattığını bul
                    for key, value in details.items():
                        try:
                            json.dumps({key: value})
                        except:
                            print(f"Sorunlu anahtar: {key}")
                            details[key] = str(value)  # String'e çevir
                
                details = convert_numpy_types(details)
                try:
                    json.dumps(details)  # JSON'a çevrilebiliyor mu test et
                except Exception as e:
                    print("JSON'a çevrilemeyen detay:", details)
                    raise e  # Hata varsa burada patlasın, neyin sorunlu olduğunu gör
                
                Analysis.objects.create(
                    owner=request.user,
                    title="Başlık",
                    summary="Özet",
                    details=details,
                    is_public=is_public

                )
                
                messages.success(request, 'Analiz başarıyla tamamlandı.')
                return redirect('analyzer:result')
            else:
                messages.error(request, 'Formda bazı hatalar var. Lütfen girişlerinizi kontrol edin.')
                return redirect('analyzer:analyze')
        except Exception as e:
            print(traceback.format_exc())  # Bu satırı ekle!
            messages.error(request, f'Analiz sırasında bir hata oluştu: {str(e)}')
            return redirect('analyzer:analyze')
    
    # GET isteği için
    if 'csv_file_path' not in request.session:
        messages.error(request, 'Lütfen önce bir CSV dosyası yükleyin.')
        return redirect('analyzer:upload')
    
    file_path = request.session['csv_file_path']
    if not os.path.exists(file_path):
        messages.error(request, 'Yüklenen dosya bulunamadı.')
        return redirect('analyzer:upload')
    
    try:
        df, _ = load_and_validate_csv(file_path)
        all_columns = df.columns.tolist()
        target_column = request.POST.get('target_column')
        if not target_column:
            target_column = guess_target_column(df)
            if not target_column:
                messages.error(request, "Hedef sütun otomatik olarak belirlenemedi. Lütfen hedef sütunu seçin.")
                return redirect('analyzer:analyze')
        
        # 🚨 Bu kısım: Hedef sütun çok fazla benzersiz değer içeriyorsa kullanıcıyı uyar
        if df[target_column].nunique() > 100 and df[target_column].dtype != 'object':
            messages.warning(request, f"Seçtiğiniz hedef sütun ('{target_column}') yüksek sayıda benzersiz değer içeriyor. Bu sütun bir sınıflandırma problemi için uygun olmayabilir.")

        target_column_choices = [(col, col) for col in all_columns]
        preprocessing_form = DataPreprocessingForm(target_column_choices=target_column_choices)

        return render(request, 'analyzer/analyze.html', {
            'preprocessing_form': preprocessing_form,
            'file_name': os.path.basename(file_path),
            'row_count': len(df),
            'column_count': len(df.columns),
        })
    except Exception as e:
        messages.error(request, str(e))
        return redirect('analyzer:upload')
@login_required(login_url='login')
def download_report(request):
    """
    PDF rapor oluşturma ve indirme
    """
    if 'analysis_results' not in request.session:
        messages.error(request, 'Analiz sonuçları bulunamadı.')
        return redirect('analyzer:upload')
    
    try:
        results = request.session['analysis_results']
        template = get_template('analyzer/report.html')
        
        context = {
            'analysis': results['analysis'],
            'plots': results.get('plots', {}),
            'preprocessing_info': results.get('preprocessing_info', {}),
            'model_metrics': results.get('model_metrics', {}),
            'model_error': results.get('model_error'),
            'file_name': results.get('file_name'),
            'timestamp': results.get('timestamp'),
            'applied_operations_summary': results.get('applied_operations_summary', {})
        }
        html = template.render(context)
        response = HttpResponse(content_type='application/pdf')
        response['Content-Disposition'] = 'attachment; filename="data_analysis_report.pdf"'
        pisa_status = pisa.CreatePDF(html, dest=response)
        if pisa_status.err:
            return HttpResponse('PDF oluşturulurken hata oluştu: %s' % pisa_status.err)
        return response
    except Exception as e:
        messages.error(request, f'PDF raporu oluşturulurken bir hata oluştu: {str(e)}')
        return redirect('analyzer:result')
@login_required(login_url='login')
def download_processed_data(request):
    """
    İşlenmiş veri setini indirme
    """
    if 'analysis_results' not in request.session or 'df_processed_data' not in request.session['analysis_results']:
        messages.error(request, 'İşlenmiş veri seti bulunamadı.')
        return redirect('analyzer:result')
    
    try:
        df_processed_json = request.session['analysis_results']['df_processed_data']
        df_processed = pd.read_json(df_processed_json, orient='split')
        
        original_file_name = os.path.basename(request.session.get('csv_file_path', 'processed_data.csv'))
        base_name, ext = os.path.splitext(original_file_name)
        response_file_name = f'{base_name}_processed.csv'

        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="{response_file_name}"'
        df_processed.to_csv(path_or_buf=response, index=False)
        return response
    except Exception as e:
        messages.error(request, f'İşlenmiş veri indirilirken bir hata oluştu: {str(e)}')
        return redirect('analyzer:result')
@login_required(login_url='login')
def show_results(request):
    """
    Analiz sonuçlarını gösterir.
    """
    if 'analysis_results' not in request.session:
        messages.error(request, 'Analiz sonuçları bulunamadı. Lütfen önce bir dosya yükleyin ve analiz yapın.')
        return redirect('analyzer:upload')  # upload sayfasına yönlendir

    results = request.session['analysis_results']
    
    return render(request, 'analyzer/result.html', {
        'analysis': results['analysis'],
        'plots': results.get('plots', {}),
        'preprocessing_info': results.get('preprocessing_info', {}),
        'model_metrics': results.get('model_metrics', {}),
        'model_error': results.get('model_error'),
        'file_name': results.get('file_name'),
        'applied_operations_summary': results.get('applied_operations_summary', {})
    })

@login_required
def save_analysis(request):
    if request.method == 'POST':
        title = request.POST.get('analysis_title')
        is_public = request.POST.get('is_public') == 'true'
        
        # Kullanıcının seçtiği kısımları al
        share_summary = request.POST.get('share_summary') == 'true'
        share_plots = request.POST.get('share_plots') == 'true'
        share_metrics = request.POST.get('share_metrics') == 'true'
        share_preprocessing = request.POST.get('share_preprocessing') == 'true'
        
        results = request.session.get('analysis_results')
        if not results:
            messages.error(request, "Kayıt edilecek analiz bulunamadı.")
            return redirect('analyzer:result')
        
        # Public ise ve hiçbir kutucuk seçilmemişse hata ver
        if is_public and not (share_summary or share_plots or share_metrics or share_preprocessing):
            messages.error(request, "Lütfen paylaşmak istediğiniz en az bir alanı seçin.")
            return render(request, 'analyzer/save_analysis.html', {
                # Gerekirse formu tekrar doldurmak için context ekle
            })
        
        # Details dict'ini oluştur
        details = {
            'target_column': results.get('target_column'),
            'model_type': results.get('model_type'),
            'short_summary': f"Veri setinde {results.get('row_count', 0)} satır ve {results.get('column_count', 0)} sütun var.",
            'file_name': results.get('file_name'),
            'row_count': results.get('row_count'),
            'column_count': results.get('column_count'),
        }
        
        if not is_public:
            details.update({
                'summary': results.get('analysis'),
                'plots': results.get('plots'),
                'model_metrics': results.get('model_metrics'),
                'preprocessing_info': results.get('preprocessing_info'),
                'applied_operations_summary': results.get('applied_operations_summary'),
            })
        else:
            if share_summary:
                details['summary'] = results.get('analysis')
            if share_plots:
                details['plots'] = results.get('plots')
            if share_metrics:
                details['model_metrics'] = results.get('model_metrics')
            if share_preprocessing:
                details['preprocessing_info'] = results.get('preprocessing_info')
        
        details = clean_json(details)
        details = convert_numpy_types(details)
        
        cover_plot_name = request.POST.get('cover_plot_name')
        cover_image = request.FILES.get('cover_image')

        # Rastgele görsel seçimi (sadece cover_image ve cover_plot_name yoksa)
        random_cover = None
        if not cover_image and not cover_plot_name:
            avatars_dir = os.path.join(settings.MEDIA_ROOT, 'avatars')
            avatar_files = [f for f in os.listdir(avatars_dir) if f.endswith('.png')]
            if avatar_files:
                random_cover = random.choice(avatar_files)

        analysis = Analysis.objects.create(
            owner=request.user,
            title=title,
            summary=results.get('short_summary', 'Özet'),
            details=details,
            is_public=is_public,
            cover_plot_name=cover_plot_name if cover_plot_name else '',
            cover_image=cover_image if cover_image else None,
            random_cover=random_cover or '',
        )
        
        messages.success(request, "Analiz başarıyla kaydedildi!")
        return redirect('analyzer:dashboard')
    
    return render(request, 'analyzer/save_analysis.html')

@login_required
def my_analyses(request):
    analyses = Analysis.objects.filter(owner=request.user).order_by('-created_at')
    paginator = Paginator(analyses, 20)  # Her sayfada 20 analiz
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    total_count = analyses.count()
    public_count = analyses.filter(is_public=True).count()
    private_count = analyses.filter(is_public=False).count()
    return render(request, 'analyzer/my_analyses.html', {
        'page_obj': page_obj,
        'total_count': total_count,
        'public_count': public_count,
        'private_count': private_count,
    })

def public_analyses(request):
    analyses = Analysis.objects.filter(is_public=True)
    user_favs = []
    other_analyses = []
    if request.user.is_authenticated:
        for analysis in analyses:
            if request.user in analysis.favorited_by.all():
                user_favs.append(analysis)
            else:
                other_analyses.append(analysis)
    else:
        other_analyses = list(analyses)
    return render(request, 'analyzer/public_analyses.html', {
        'user_favs': user_favs,
        'other_analyses': other_analyses,
        'analyses': analyses,  # İstatistikler için
    })

@login_required
def edit_analysis(request, analysis_id):
    analysis = get_object_or_404(Analysis, id=analysis_id, owner=request.user)
    if request.method == "POST":
        title = request.POST.get("title")
        summary = request.POST.get("summary")
        is_public = request.POST.get("is_public") == "true"
        
        # Kullanıcının seçtiği kısımları al
        share_summary = request.POST.get('share_summary') == 'true'
        share_plots = request.POST.get('share_plots') == 'true'
        share_metrics = request.POST.get('share_metrics') == 'true'
        share_preprocessing = request.POST.get('share_preprocessing') == 'true'
        
        # Eski detayları al
        old_details = analysis.details.copy()
        details = {
            'target_column': old_details.get('target_column'),
            'model_type': old_details.get('model_type'),
            'short_summary': old_details.get('short_summary'),
            'file_name': old_details.get('file_name'),
            'row_count': old_details.get('row_count'),
            'column_count': old_details.get('column_count'),
        }
        
        if is_public:
            # Sadece seçilen kısımları ekle
            if share_summary and 'summary' in old_details:
                details['summary'] = old_details['summary']
            if share_plots and 'plots' in old_details:
                details['plots'] = old_details['plots']
            if share_metrics and 'model_metrics' in old_details:
                details['model_metrics'] = old_details['model_metrics']
            if share_preprocessing and 'preprocessing_info' in old_details:
                details['preprocessing_info'] = old_details['preprocessing_info']
        else:
            # Private ise tüm detayları ekle
            details.update({
                'summary': old_details.get('summary'),
                'plots': old_details.get('plots'),
                'model_metrics': old_details.get('model_metrics'),
                'preprocessing_info': old_details.get('preprocessing_info'),
                'applied_operations_summary': old_details.get('applied_operations_summary'),
            })
        
        details = clean_json(details)
        details = convert_numpy_types(details)
        analysis.title = title
        analysis.summary = summary
        analysis.is_public = is_public
        analysis.details = details
        analysis.save()
        
        messages.success(request, "Analiz başarıyla güncellendi.")
        return redirect('analyzer:dashboard')
    
    return render(request, "analyzer/edit_analysis.html", {"analysis": analysis})

 # Sadece test için! (CSRF token düzgün çalışıyorsa kaldır)
@login_required
def delete_analysis(request, analysis_id):
    print("Silme view'ı çalıştı", file=sys.stderr)
    analysis = get_object_or_404(Analysis, id=analysis_id, owner=request.user)
    print(f"Analiz bulundu: {analysis.id}", file=sys.stderr)
    if request.method == "POST":
        print(f"Siliniyor: {analysis.id}", file=sys.stderr)
        analysis.delete()
        print("Silindi!", file=sys.stderr)
        if request.headers.get('x-requested-with') == 'XMLHttpRequest':
            return JsonResponse({'success': True})
        messages.success(request, "Analiz silindi.")
        return redirect('analyzer:dashboard')
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        return JsonResponse({'error': 'Yalnızca POST ile silinebilir.'}, status=400)
    return render(request, "analyzer/delete_analysis_confirm.html", {"analysis": analysis})

@login_required
def toggle_favorite(request, analysis_id):
    analysis = get_object_or_404(Analysis, id=analysis_id, is_public=True)
    if request.user in analysis.favorited_by.all():
        analysis.favorited_by.remove(request.user)
        favorited = False
    else:
        analysis.favorited_by.add(request.user)
        favorited = True
    return JsonResponse({'favorited': favorited})

@csrf_exempt
@login_required
def upload_example(request):
    file_name = request.GET.get('file')
    example_path = os.path.join(settings.MEDIA_ROOT, file_name)
    if not os.path.exists(example_path):
        return JsonResponse({'success': False, 'error': 'Dosya bulunamadı.'})
    user_file_path = os.path.join(settings.MEDIA_ROOT, f"{request.user.username}_{file_name}")
    shutil.copy(example_path, user_file_path)
    request.session['csv_file_path'] = user_file_path
    return JsonResponse({'success': True})

def faq(request):
    return render(request, 'analyzer/faq.html')
