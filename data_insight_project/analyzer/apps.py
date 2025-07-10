from django.apps import AppConfig


class AnalyzerConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'analyzer'

    def ready(self):
        # Bu metod Django uygulaması başladığında çağrılır.
        # Şablon etiketlerini açıkça yükleyerek kayıtlarını sağlamak için
        # burada bir import denemesi yapıyoruz.
        try:
            import analyzer.templatetags.analyzer_filters # Şablon etiketlerini yükle
        except ImportError:
            pass
