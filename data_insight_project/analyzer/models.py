from django.db import models
from django.contrib.auth.models import User

def user_avatar_path(instance, filename):
    # Her kullanıcıya özel klasör
    return f'avatars/user_{instance.user.id}/{filename}'

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    birth_date = models.DateField(null=True, blank=True)
    gender = models.CharField(max_length=10, choices=(
        ('male', 'Erkek'),
        ('female', 'Kadın'),
        ('other', 'Diğer'),
    ), blank=True)
    accept_terms = models.BooleanField(default=False)
    # avatar ve avatar_choice alanları kaldırıldı

    # get_avatar_url fonksiyonu kaldırıldı

def analysis_cover_path(instance, filename):
    return f'analysis_covers/{instance.owner.id}/{filename}'

class Analysis(models.Model):
    owner = models.ForeignKey(User, on_delete=models.CASCADE)
    title = models.CharField(max_length=100)
    summary = models.TextField(blank=True)
    details = models.JSONField(default=dict, blank=True, null=True)
    is_public = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    favorited_by = models.ManyToManyField(User, related_name='favorite_analyses', blank=True)
    cover_image = models.ImageField(upload_to=analysis_cover_path, null=True, blank=True)
    cover_plot_name = models.CharField(max_length=100, blank=True)
    random_cover = models.CharField(max_length=100, blank=True, null=True)

    def __str__(self):
        return self.title