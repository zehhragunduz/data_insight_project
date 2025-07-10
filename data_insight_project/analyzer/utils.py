import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Tkinter hatalarını önlemek için backend'i Agg olarak ayarla
import matplotlib.pyplot as plt
import seaborn as sns
import logging # logging modülünü import et

# Logger yapılandırması
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Console handler ekle (eğer zaten ekli değilse)
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, classification_report, mean_squared_error, r2_score,
    mean_absolute_error, mean_absolute_percentage_error,roc_curve, auc,
    precision_recall_curve
)
from sklearn.impute import KNNImputer
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import io
import base64
from typing import Tuple, Dict, List, Optional
import warnings
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import learning_curve
warnings.filterwarnings('ignore')

import csv

def load_and_validate_csv(file_path: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    CSV dosyasını yükler ve temel doğrulama yapar.
    """
    logger.info(f"CSV dosyası yükleniyor: {file_path}")
    try:
        # Önce ilk satırı oku ve ayraç tahmini yap
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            sample = f.read(2048)
            sniffer = csv.Sniffer()
            try:
                delimiter = sniffer.sniff(sample).delimiter
                logger.info(f"Algılanan delimiter: {delimiter}")
            except Exception:
                # Otomatik algılanamazsa önce ; sonra , dene
                logger.warning("Delimiter otomatik algılanamadı, fallback olarak ; ve , denenecek.")
                try:
                    # Test için küçük bir örnekle ayırmayı dene
                    pd.read_csv(file_path, delimiter=';', encoding='utf-8', nrows=1)
                    delimiter = ';'
                except Exception:
                    delimiter = ','
                logger.info(f"Fallback delimiter: {delimiter}")

        try:
            df = pd.read_csv(file_path, delimiter=delimiter, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, delimiter=delimiter, encoding='ISO-8859-9')

        warnings = []

        # Veri tipi uyarıları
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    pd.to_numeric(df[col])
                    warnings.append(f"'{col}' sütunu sayısal değerler içeriyor ama object tipinde.")
                except:
                    pass

        logger.info("CSV dosyası başarıyla yüklendi.")
        return df, warnings

    except Exception as e:
        logger.error(f"CSV dosyası yüklenirken hata oluştu: {str(e)}")
        raise ValueError(f"CSV dosyası yüklenirken hata oluştu: {str(e)}")

    

def analyze_data(df: pd.DataFrame) -> Dict:
    """
    Veri seti üzerinde temel analizler yapar.
    """
    logger.info("Veri analizi başlatıldı.")
    
    unique_value_info = {}
    for col in df.columns:
        unique_counts = df[col].value_counts()
        total_rows = len(df[col].dropna()) if df[col].dtype == 'object' else len(df) # NaNları düşerek say
        unique_count = unique_counts.shape[0]
        
        percentage = (unique_count / total_rows * 100) if total_rows > 0 else 0
        top_5_values = unique_counts.head(5).to_dict()
        
        unique_value_info[col] = {
            'count': int(unique_count),
            'percentage': float(f'{percentage:.2f}'),
            'top_5_values': {str(k): int(v) for k, v in top_5_values.items()} # K-V'leri string ve int yap
        }

    analysis = {
        'summary': df.describe().to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentages': (df.isnull().sum() / len(df) * 100).to_dict(),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'constant_columns': df.columns[df.nunique() == 1].tolist(),
        'high_missing_columns': df.columns[df.isnull().sum() / len(df) > 0.8].tolist(),
        'duplicate_rows': int(df.duplicated().sum()),
        'total_rows': int(len(df)),
        'total_columns': int(len(df.columns)),
        'unique_value_info': unique_value_info # Yeni formatta ekle
    }
    logger.info("Veri analizi tamamlandı.")
    return analysis

def generate_plots(
    df: pd.DataFrame,
    y_test: Optional[List[float]] = None,
    y_pred: Optional[List[float]] = None,
    is_classification: Optional[bool] = None,
    importances: Optional[List[float]] = None,
    feature_names: Optional[List[str]] = None,
    model=None,
    X=None,
    y=None
) -> Dict[str, str]:

    """
    Veri seti ve model sonuçları için otomatik grafikler oluşturur.
    """
    logger.info("Grafik oluşturma başlatıldı.")
    plots = {}
    
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        logger.info(f"Sayısal sütunlar: {numeric_cols.tolist()}")

        if len(numeric_cols) > 0:
            for col in numeric_cols:
                try:
                    fig = plt.figure(figsize=(10, 6))
                    plt.hist(df[col].dropna(), bins=30)
                    plt.title(f'{col} Histogramı')
                    plt.xlabel(col)
                    plt.ylabel('Frekans')
                    plt.tight_layout()
                    plot_data = fig_to_base64(fig)
                    if plot_data:
                        plots[f'histogram_{col}'] = plot_data
                        logger.info(f"{col} histogramı başarıyla oluşturuldu.")
                    else:
                        logger.warning(f"{col} histogramı base64 dönüşümünde başarısız oldu.")
                    plt.close(fig)
                except Exception as e:
                    logger.error(f"{col} histogramı oluşturulurken hata: {str(e)}")
                    if 'fig' in locals() and plt.fignum_exists(fig.number):
                        plt.close(fig)
        
        if len(numeric_cols) > 0:
            try:
                fig = plt.figure(figsize=(12, 6))
                df[numeric_cols].boxplot()
                plt.title('Sayısal Değişkenler Box Plot')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plot_data = fig_to_base64(fig)
                if plot_data:
                    plots['boxplot'] = plot_data
                    logger.info("Box plot başarıyla oluşturuldu.")
                else:
                    logger.warning("Box plot base64 dönüşümünde başarısız oldu.")
                plt.close(fig)
            except Exception as e:
                logger.error(f"Box plot oluşturulurken hata: {str(e)}")
                if 'fig' in locals() and plt.fignum_exists(fig.number):
                    plt.close(fig)
        
        if len(numeric_cols) > 1:
            try:
                fig = plt.figure(figsize=(10, 8))
                corr_matrix = df[numeric_cols].corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
                plt.title('Korelasyon Matrisi')
                plt.tight_layout()
                plot_data = fig_to_base64(fig)
                if plot_data:
                    plots['correlation'] = plot_data
                    logger.info("Korelasyon matrisi başarıyla oluşturuldu.")
                else:
                    logger.warning("Korelasyon matrisi base64 dönüşümünde başarısız oldu.")
                plt.close(fig)
            except Exception as e:
                logger.error(f"Korelasyon matrisi oluşturulurken hata: {str(e)}")
                if 'fig' in locals() and plt.fignum_exists(fig.number):
                    plt.close(fig)
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        logger.info(f"Kategorik sütunlar: {categorical_cols.tolist()}")

        for col in categorical_cols:
            if df[col].nunique() <= 10:
                try:
                    fig = plt.figure(figsize=(10, 6))
                    value_counts = df[col].value_counts()
                    value_counts.plot(kind='bar')
                    plt.title(f'{col} Değer Dağılımı')
                    plt.xlabel(col)
                    plt.ylabel('Frekans')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plot_data = fig_to_base64(fig)
                    if plot_data:
                        plots[f'barchart_{col}'] = plot_data
                        logger.info(f"{col} bar chart başarıyla oluşturuldu.")
                    else:
                        logger.warning(f"{col} bar chart base64 dönüşümünde başarısız oldu.")
                    plt.close(fig)
                except Exception as e:
                    logger.error(f"{col} bar chart oluşturulurken hata: {str(e)}")
                    if 'fig' in locals() and plt.fignum_exists(fig.number):
                        plt.close(fig)

        # Model Değerlendirme Grafikleri
        if y_test is not None and y_pred is not None:
            y_test_np = np.array(y_test)
            y_pred_np = np.array(y_pred)
            
            if is_classification:
                logger.info("Sınıflandırma model değerlendirme grafikleri oluşturuluyor.")
                # Confusion Matrix
                try:
                    cm = confusion_matrix(y_test_np, y_pred_np)
                    fig_cm = plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                    plt.title('Confusion Matrix')
                    plt.xlabel('Tahmin Edilen Sınıf')
                    plt.ylabel('Gerçek Sınıf')
                    plt.tight_layout()
                    plot_data_cm = fig_to_base64(fig_cm)
                    if plot_data_cm: plots['confusion_matrix'] = plot_data_cm
                    plt.close(fig_cm)
                    logger.info("Confusion Matrix başarıyla oluşturuldu.")
                except Exception as e:
                    logger.error(f"Confusion Matrix oluşturulurken hata: {str(e)}")

                # ROC Eğrisi (Binary Classification için)
                # y_test ve y_pred'in ikili sınıflandırma için uygun olduğundan emin olun
                if len(np.unique(y_test_np)) == 2 and hasattr(y_pred_np, 'ndim') and y_pred_np.ndim == 1: # y_pred 1D ise (yani sınıf etiketleri)
                    try:
                        # Eğer y_pred direkt sınıf etiketleri ise, ROC için olasılıklara ihtiyacımız var.
                        # Bu durumda ROC çizmek için modelden olasılıkları almamız gerekirdi, ki bu şu anki yapıda yok.
                        # Basit bir ROC çizimi için, eğer y_pred 0 veya 1 ise doğrudan kullanabiliriz.
                        # Daha doğru bir yaklaşım için modelin predict_proba'sına ihtiyacımız olur.
                        fpr, tpr, thresholds = roc_curve(y_test_np, y_pred_np)
                        roc_auc = auc(fpr, tpr)
                        fig_roc = plt.figure(figsize=(8, 6))
                        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                        plt.xlim([0.0, 1.0])
                        plt.ylim([0.0, 1.05])
                        plt.xlabel('False Positive Rate')
                        plt.ylabel('True Positive Rate')
                        plt.title('Receiver Operating Characteristic')
                        plt.legend(loc="lower right")
                        plt.tight_layout()
                        plot_data_roc = fig_to_base64(fig_roc)
                        if plot_data_roc: plots['roc_curve'] = plot_data_roc
                        plt.close(fig_roc)
                        logger.info("ROC Eğrisi başarıyla oluşturuldu.")
                    except Exception as e:
                        logger.error(f"ROC Eğrisi oluşturulurken hata: {str(e)}")

            else: # Regresyon
                logger.info("Regresyon model değerlendirme grafikleri oluşturuluyor.")
                # Gerçek ve Tahmin Edilen Değerler Grafiği
                try:
                    fig_scatter = plt.figure(figsize=(10, 8))
                    plt.scatter(y_test_np, y_pred_np, alpha=0.6)
                    plt.plot([y_test_np.min(), y_test_np.max()], [y_test_np.min(), y_test_np.max()], '--r', lw=2)
                    plt.title('Gerçek ve Tahmin Edilen Değerler')
                    plt.xlabel('Gerçek Değerler')
                    plt.ylabel('Tahmin Edilen Değerler')
                    plt.grid(True)
                    plt.tight_layout()
                    plot_data_scatter = fig_to_base64(fig_scatter)
                    if plot_data_scatter: plots['actual_vs_predicted'] = plot_data_scatter
                    plt.close(fig_scatter)
                    logger.info("Gerçek ve Tahmin Edilen Değerler Grafiği başarıyla oluşturuldu.")
                except Exception as e:
                    logger.error(f"Gerçek ve Tahmin Edilen Değerler Grafiği oluşturulurken hata: {str(e)}")

                # Artık (Residual) Grafiği
                try:
                    residuals = y_test_np - y_pred_np
                    fig_residual = plt.figure(figsize=(10, 6))
                    plt.scatter(y_pred_np, residuals, alpha=0.6)
                    plt.hlines(0, xmin=y_pred_np.min(), xmax=y_pred_np.max(), colors='r', linestyles='--')
                    plt.title('Artıklar Grafiği')
                    plt.xlabel('Tahmin Edilen Değerler')
                    plt.ylabel('Artıklar')
                    plt.grid(True)
                    plt.tight_layout()
                    plot_data_residual = fig_to_base64(fig_residual)
                    if plot_data_residual: plots['residuals_plot'] = plot_data_residual
                    plt.close(fig_residual)
                    logger.info("Artıklar Grafiği başarıyla oluşturuldu.")
                except Exception as e:
                    logger.error(f"Artıklar Grafiği oluşturulurken hata: {str(e)}")

                                # Özellik Önem Grafiği
                
                    try:
                        logger.info("Feature importance grafiği oluşturuluyor.")
                        importances_np = np.array(importances)
                        feature_names_np = np.array(feature_names)
                        indices = np.argsort(importances_np)[::-1]
                        
                        fig_importance = plt.figure(figsize=(10, 6))
                        plt.bar(range(len(importances_np)), importances_np[indices], align='center')
                        plt.xticks(range(len(importances_np)), feature_names_np[indices], rotation=90)
                        plt.title('Özellik Önem Dereceleri (Feature Importance)')
                        plt.tight_layout()
                        
                        plot_data_importance = fig_to_base64(fig_importance)
                        if plot_data_importance:
                            plots['feature_importance'] = plot_data_importance
                            logger.info("Feature importance grafiği başarıyla oluşturuldu.")
                        else:
                            logger.warning("Feature importance base64 dönüşümünde başarısız oldu.")
                        plt.close(fig_importance)
                    except Exception as e:
                        logger.error(f"Feature importance grafiği oluşturulurken hata: {str(e)}")

        # Precision-Recall Curve (sadece sınıflandırma için)
        if is_classification and y_test is not None and y_pred is not None:
            try:
                if hasattr(model, "predict_proba"):
                    y_score = model.predict_proba(X)[:, 1] if model.predict_proba(X).shape[1] > 1 else model.predict_proba(X)[:, 0]
                else:
                    y_score = y_pred
                precision, recall, _ = precision_recall_curve(y_test, y_score)
                fig_pr = plt.figure(figsize=(8, 6))
                plt.plot(recall, precision, color='b', lw=2)
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Precision-Recall Curve')
                plt.tight_layout()
                plot_data_pr = fig_to_base64(fig_pr)
                if plot_data_pr:
                    plots['precision_recall_curve'] = plot_data_pr
                plt.close(fig_pr)
            except Exception as e:
                logger.error(f"Precision-Recall Curve oluşturulurken hata: {str(e)}")
        # PCA Scatter Plot
        try:
            if df.select_dtypes(include=[np.number]).shape[1] >= 2:
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(df.select_dtypes(include=[np.number]))
                fig_pca = plt.figure(figsize=(8, 6))
                if y is not None:
                    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
                    plt.colorbar(scatter)
                else:
                    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)
                plt.title('PCA Scatter Plot')
                plt.xlabel('PC1')
                plt.ylabel('PC2')
                plt.tight_layout()
                plot_data_pca = fig_to_base64(fig_pca)
                if plot_data_pca:
                    plots['pca_scatter'] = plot_data_pca
                plt.close(fig_pca)
        except Exception as e:
            logger.error(f"PCA Scatter oluşturulurken hata: {str(e)}")
        # t-SNE Scatter Plot
        try:
            if df.select_dtypes(include=[np.number]).shape[1] >= 2 and df.shape[0] > 10:
                tsne = TSNE(n_components=2, random_state=42, perplexity=5)
                X_tsne = tsne.fit_transform(df.select_dtypes(include=[np.number]))
                fig_tsne = plt.figure(figsize=(8, 6))
                if y is not None:
                    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.7)
                    plt.colorbar(scatter)
                else:
                    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.7)
                plt.title('t-SNE Scatter Plot')
                plt.xlabel('Dim 1')
                plt.ylabel('Dim 2')
                plt.tight_layout()
                plot_data_tsne = fig_to_base64(fig_tsne)
                if plot_data_tsne:
                    plots['tsne_scatter'] = plot_data_tsne
                plt.close(fig_tsne)
        except Exception as e:
            logger.error(f"t-SNE Scatter oluşturulurken hata: {str(e)}")
        # Learning Curve
        try:
            if model is not None and X is not None and y is not None:
                train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=3, n_jobs=1, train_sizes=np.linspace(0.1, 1.0, 5))
                train_scores_mean = np.mean(train_scores, axis=1)
                test_scores_mean = np.mean(test_scores, axis=1)
                fig_lc = plt.figure(figsize=(8, 6))
                plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Eğitim Skoru')
                plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Doğrulama Skoru')
                plt.title('Learning Curve')
                plt.xlabel('Eğitim Seti Boyutu')
                plt.ylabel('Skor')
                plt.legend(loc='best')
                plt.tight_layout()
                plot_data_lc = fig_to_base64(fig_lc)
                if plot_data_lc:
                    plots['learning_curve'] = plot_data_lc
                plt.close(fig_lc)
        except Exception as e:
            logger.error(f"Learning Curve oluşturulurken hata: {str(e)}")

    except Exception as e:
        logger.critical(f"Genel grafik oluşturma hatası: {str(e)}")
        return {}
    
    logger.info("Grafik oluşturma tamamlandı.")
    return plots







def preprocess_data(df: pd.DataFrame, options: Dict) -> Tuple[pd.DataFrame, Dict]:
    """
    Veri ön işleme adımlarını uygular.
    """
    logger.info("Veri ön işleme başlatıldı.")
    preprocessing_info = {}
    df_processed = df.copy()
    
    # Sabit sütunları kaldır
    if options.get('remove_constant'):
        constant_cols = df_processed.columns[df_processed.nunique() == 1].tolist()
        if constant_cols:
            df_processed = df_processed.drop(columns=constant_cols)
            preprocessing_info['removed_constant_columns'] = constant_cols
            logger.info(f"Sabit sütunlar kaldırıldı: {constant_cols}")
    
    # Yüksek eksik veri içeren sütunları kaldır
    if options.get('remove_high_missing'):
        high_missing_cols = df_processed.columns[df_processed.isnull().sum() / len(df_processed) > 0.8].tolist()
        if high_missing_cols:
            df_processed = df_processed.drop(columns=high_missing_cols)
            preprocessing_info['removed_high_missing_columns'] = high_missing_cols
            logger.info(f"Yüksek eksik veri içeren sütunlar kaldırıldı: {high_missing_cols}")
    
    # Eksik veri doldurma
    fill_method = options.get('fill_missing', 'mean')
    
    if fill_method != 'none':
        logger.info(f"Eksik veri doldurma yöntemi: {fill_method}")
        if fill_method == 'knn':
        # Sadece sayısal sütunlarda uygula
            numeric_cols = df_processed.select_dtypes(include=['float64', 'int64']).columns
            n_neighbors = options.get('knn_n_neighbors', 3)
            if len(df_processed) < n_neighbors:
                n_neighbors = len(df_processed)
            imputer = KNNImputer(n_neighbors=n_neighbors)
            df_processed[numeric_cols] = imputer.fit_transform(df_processed[numeric_cols])
            logger.info("Eksik veriler KNN Imputer ile dolduruldu.")
        # KNN ile doldurulan hücre sayısını hesapla (isteğe bağlı)
        # filled_count = ... (dilersen ekleyebilirsin)
        else:
            for col in df_processed.columns:
                if df_processed[col].isnull().any():
                    original_missing_count = df_processed[col].isnull().sum()
                    if df_processed[col].dtype in ['int64', 'float64']:
                        if fill_method == 'mean':
                            df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
                        elif fill_method == 'median':
                            df_processed[col] = df_processed[col].fillna(df_processed[col].median())
                    else:
                        if fill_method == 'mode':
                            mode_val = df_processed[col].mode()
                            if not mode_val.empty:
                                df_processed[col] = df_processed[col].fillna(mode_val[0])
                            else:
                                df_processed[col] = df_processed[col].fillna("Bilinmiyor")
                        elif fill_method == 'drop':
                            df_processed = df_processed.dropna(subset=[col])
                    filled_count = original_missing_count - df_processed[col].isnull().sum()
                    if filled_count > 0:
                        if 'filled_missing_cells' not in preprocessing_info:
                            preprocessing_info['filled_missing_cells'] = 0
                        preprocessing_info['filled_missing_cells'] += filled_count
                        logger.info(f"{col} sütununda {filled_count} eksik veri dolduruldu.")

    # Kategorik veri dönüşümü
    encoding_method = options.get('encoding_method', 'none')
    if encoding_method != 'none':
        logger.info(f"Kategorik veri dönüşüm yöntemi: {encoding_method}")
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            preprocessing_info['categorical_encoding_method'] = encoding_method
            preprocessing_info['encoded_columns'] = categorical_cols.tolist()
            if encoding_method == 'label':
                for col in categorical_cols:
                    le = LabelEncoder()
                    df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                    logger.info(f"{col} sütunu Label Encoding ile dönüştürüldü.")
            elif encoding_method == 'onehot':
                df_processed = pd.get_dummies(df_processed, columns=categorical_cols)
                logger.info(f"Kategorik sütunlar One-Hot Encoding ile dönüştürüldü.")
    logger.info("Veri ön işleme tamamlandı.")
    # ... preprocess_data fonksiyonunun sonunda, return'den önce:
    if options.get('feature_selection', False):
        target_col = options.get('target_col')  # Hedef değişkenin adı
        n_features = options.get('n_features', 5)
        if target_col and target_col in df_processed.columns:
            X = df_processed.drop(columns=[target_col])
            y = df_processed[target_col]
            selector = RFE(RandomForestClassifier(), n_features_to_select=n_features)
            selector.fit(X, y)
            selected_columns = X.columns[selector.support_]
        # Sadece seçilen sütunları ve hedefi bırak
            df_processed = df_processed[selected_columns.tolist() + [target_col]]
            preprocessing_info['selected_features'] = selected_columns.tolist()
    # ... feature selection kodundan hemen sonra, return'den önce ekle:
    if df_processed.isnull().any().any():
        logger.warning("Veri ön işleme sonunda hala eksik veri (NaN) var. Tüm satırlar siliniyor.")
        df_processed = df_processed.dropna()
    return df_processed, preprocessing_info

def train_model(df: pd.DataFrame, target_column: str, model_type: str, options: Dict) -> Tuple[Dict, Optional[str]]:
    """
    Seçilen model tipine göre makine öğrenimi modeli eğitir ve metrikleri döndürür.
    """
    logger.info(f"Model eğitimi başlatıldı. Model Tipi: {model_type}, Hedef Sütun: {target_column}")
    metrics = {}
    error = None
    is_classification = False
  
    if target_column not in df.columns:
        error = f"Hedef sütun '{target_column}' veri setinde bulunamadı."
        logger.error(error)
        return {}, error,{}
    
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Eğitim verisine SMOTE uygula
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if is_classification:
        smote = SMOTE()
        X_train, y_train = smote.fit_resample(X_train, y_train)

    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        logger.info(f"{col} özelliği Label Encoding ile dönüştürüldü (model eğitimi için).")

    for col in X.columns:
        if X[col].dtype == 'object':
            error = f"'train_model' fonksiyonunda hata: '{col}' sütunu hala object tipinde. Lütfen ön işleme adımlarını kontrol edin."
            logger.error(error)
            return {}, error,{}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info("Veri eğitim ve test setlerine ayrıldı.")

    model = None
    is_classification = False

    if y.nunique() <= 20 and y.dtype == 'object':
         is_classification = True
         le_y = LabelEncoder()
         y_train = le_y.fit_transform(y_train)
         y_test = le_y.transform(y_test)
         logger.info("Hedef sütun kategorik olarak belirlendi (object tipinde ve <=20 benzersiz değer).")
    elif y.nunique() <= 20 and y.dtype != 'object':
        is_classification = True
        logger.info("Hedef sütun sayısal ama az sayıda benzersiz değer içerdiğinden sınıflandırma olarak belirlendi.")
    else:
        is_classification = False
        logger.info("Hedef sütun regresyon olarak belirlendi.")

    try:
        if is_classification:
            if model_type == 'logistic':
                model = LogisticRegression(max_iter=1000)
                logger.info("Lojistik Regresyon modeli seçildi.")
            elif model_type == 'decision_tree':
                max_depth = options.get('dt_max_depth', None)
                model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
                logger.info(f"Karar Ağacı sınıflandırma modeli seçildi (Max Depth: {max_depth}).")
            elif model_type == 'random_forest':
                n_estimators = options.get('rf_n_estimators', 100)
                model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
                logger.info(f"Random Forest sınıflandırma modeli seçildi (N Estimators: {n_estimators}).")
            elif model_type == 'svm':
                model = SVC(probability=True, random_state=42)
                logger.info("Destek Vektör Makinesi sınıflandırma modeli seçildi.")
            elif model_type == 'knn':
                n_neighbors = options.get('knn_n_neighbors', 3)
                
                if len(X_train) < n_neighbors:
                    n_neighbors = len(X_train)
                    logger.warning(f"KNN için n_neighbors, eğitim veri sayısından büyük. n_neighbors={n_neighbors} olarak ayarlandı.")
                model = KNeighborsClassifier(n_neighbors=n_neighbors)
                logger.info(f"K-En Yakın Komşu sınıflandırma modeli seçildi (N Neighbors: {n_neighbors}).")
            elif model_type == 'naive_bayes':
                model = GaussianNB()
                logger.info("Naive Bayes sınıflandırma modeli seçildi.")
            elif model_type == 'xgboost':
                model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
                logger.info("XGBoost sınıflandırma modeli seçildi.")
            elif model_type == 'lightgbm':
                model = LGBMClassifier()
                logger.info("LightGBM sınıflandırma modeli seçildi.")
            else:
                error = "Geçersiz sınıflandırma modeli seçimi."
                logger.error(error)
        else:
            if model_type == 'logistic': # Regresyon için varsayılan olarak Linear Regression kullan
                model = LinearRegression()
                logger.info("Lojistik Regresyon seçildi ama hedef regresyon, bu yüzden Linear Regression kullanılıyor.")
            elif model_type == 'decision_tree':
                max_depth = options.get('dt_max_depth', None)
                model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
                logger.info(f"Karar Ağacı regresyon modeli seçildi (Max Depth: {max_depth}).")
            elif model_type == 'random_forest':
                n_estimators = options.get('rf_n_estimators', 100)
                model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
                logger.info(f"Random Forest regresyon modeli seçildi (N Estimators: {n_estimators}).")
            elif model_type == 'svm':
                model = SVR()
                logger.info("Destek Vektör Makinesi regresyon modeli seçildi.")
            elif model_type == 'knn':
                n_neighbors = options.get('knn_n_neighbors', 3)
                if len(X_train) < n_neighbors:
                    n_neighbors = len(X_train)
                    logger.warning(f"KNN için n_neighbors, eğitim veri sayısından büyük. n_neighbors={n_neighbors} olarak ayarlandı.")
                model = KNeighborsClassifier(n_neighbors=n_neighbors)
                logger.info(f"K-En Yakın Komşu regresyon modeli seçildi (N Neighbors: {n_neighbors}).")
            elif model_type == 'ridge':
                model = Ridge()
                logger.info("Ridge regresyon modeli seçildi.")
            elif model_type == 'lasso':
                model = Lasso()
                logger.info("Lasso regresyon modeli seçildi.")
            elif model_type == 'xgboost':
                model = XGBRegressor()
                logger.info("XGBoost regresyon modeli seçildi.")
            elif model_type == 'lightgbm':
                model = LGBMRegressor()
                logger.info("LightGBM regresyon modeli seçildi.")
            else:
                error = "Geçersiz regresyon modeli seçimi."
                logger.error(error)

        if model:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            logger.info("Model başarıyla eğitildi ve tahminler yapıldı.")
            
            if is_classification:
                metrics['accuracy'] = accuracy_score(y_test, y_pred)
                metrics['f1_score'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                metrics['precision'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                metrics['recall'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
                metrics['classification_report'] = classification_report(y_test, y_pred, output_dict=True)
                try:
                    if hasattr(model, "predict_proba"):
                        y_score = model.predict_proba(X_test)
                        if y_score.shape[1] == 2:  # Binary classification
                            fpr, tpr, thresholds = roc_curve(y_test, y_score[:, 1])
                            metrics['roc_curve'] = {
                                'fpr': fpr.tolist(),
                                'tpr': tpr.tolist(),
                                'thresholds': thresholds.tolist()
                            }
                            metrics['auc'] = auc(fpr, tpr)
                            metrics['roc_auc'] = roc_auc_score(y_test, y_score[:, 1])
                        else:  # Multi-class
                            metrics['roc_curve'] = {}
                            metrics['auc'] = {}
                            for i in range(y_score.shape[1]):
                                fpr, tpr, thresholds = roc_curve(y_test == i, y_score[:, i])
                                metrics['roc_curve'][str(i)] = {
                                    'fpr': fpr.tolist(),
                                    'tpr': tpr.tolist(),
                                    'thresholds': thresholds.tolist()
                                }
                                metrics['auc'][str(i)] = auc(fpr, tpr)
                            metrics['roc_auc'] = roc_auc_score(y_test, y_score, multi_class='ovr', average='weighted')
                except Exception as e:
                    metrics['roc_curve'] = None
                    metrics['auc'] = None
                    metrics['roc_auc'] = None
                logger.info(
                    f"Sınıflandırma Metrikleri: "
                    f"Accuracy={metrics['accuracy']:.4f}, "
                    f"F1-Score={metrics['f1_score']:.4f}, "
                    f"Precision={metrics['precision']:.4f}, "
                    f"Recall={metrics['recall']:.4f}, "
                    f"ROC-AUC={metrics.get('roc_auc', None)}"
                )
            else:
                metrics['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
                metrics['r2_score'] = r2_score(y_test, y_pred)
                metrics['mae'] = mean_absolute_error(y_test, y_pred)
                try:
                    metrics['mape'] = mean_absolute_percentage_error(y_test, y_pred)
                except Exception:
                    metrics['mape'] = None
                logger.info(
                    f"Regresyon Metrikleri: "
                    f"RMSE={metrics['rmse']:.4f}, "
                    f"R2-Score={metrics['r2_score']:.4f}, "
                    f"MAE={metrics['mae']:.4f}, "
                    f"MAPE={metrics.get('mape', None)}"
                )



            # Özellik önemliliği ve katsayıları kısmı aynen kalabilir
            if hasattr(model, 'feature_importances_'):
                feature_importances = dict(zip(X.columns, model.feature_importances_))
                metrics['feature_importances'] = {k: float(v) for k, v in sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)}
                logger.info("Özellik önemliliği hesaplandı.")
            elif hasattr(model, 'coef_') and len(model.coef_) == len(X.columns):
                feature_coefficients = dict(zip(X.columns, model.coef_))
                metrics['feature_coefficients'] = {k: float(v) for k, v in sorted(feature_coefficients.items(), key=lambda item: abs(item[1]), reverse=True)}
                logger.info("Özellik katsayıları hesaplandı.")
            
      

    except Exception as e:
        error = f"Model eğitimi sırasında hata oluştu: {str(e)}"
        logger.critical(error)
        return {}, error,{}
    plots = generate_plots(
        df,  # df_processed yerine df!
        y_test=y_test,
        y_pred=y_pred,
        is_classification=is_classification,
        feature_names=X_test.columns if X_test is not None else None,
        model=model,
        X=X_test,
        y=y_test
    )
    logger.info("Model eğitimi tamamlandı.")
    return metrics, error, plots

def guess_target_column(df):
    # 1. En yaygın hedef isimlerini kontrol et
    target_names = ['target', 'label', 'class', 'outcome', 'survived', 'y']
    for name in target_names:
        for col in df.columns:
            if col.strip().lower() == name:
                return col
    # 2. PassengerId, Name, id gibi anlamsızları çıkar, son sütunu hedef olarak al
    exclude = ['id', 'name', 'passengerid', 'time']
    candidates = [col for col in df.columns if col.strip().lower() not in exclude]
    if candidates:
        return candidates[-1]  # Son sütun
    # 3. Hiçbiri yoksa None döndür
    return None

def fig_to_base64(fig) -> str:
    """
    Matplotlib figürünü base64 formatına dönüştürür.
    """
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        logger.info("Grafik base64 formatına başarıyla dönüştürüldü.")
        return img_str
    except Exception as e:
        logger.error(f"Base64 dönüşüm hatası: {str(e)}")
        return "" 