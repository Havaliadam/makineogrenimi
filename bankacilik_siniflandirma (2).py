"""
ÖDEV: Türkiye 2026 Bankacılık Verisi ile Hesap Aktiflik Tahmini

Amaç:
Bu projede banka müşterilerinin hesabının aktif olup olmadığını (Hesap_Aktif)
tahmin eden bir sınıflandırma modeli geliştirilmiştir. Veri ön işleme,
öznitelik üretimi, eğitim/doğrulama/test ayrımı ve iki farklı makine öğrenmesi
modelinin karşılaştırılması amaçlanmaktadır.

Kullanılan kütüphaneler:
- pandas, numpy: veri okuma ve veri işleme
- scikit-learn: veri bölme, ön işleme, modelleme ve değerlendirme
- matplotlib, seaborn: confusion matrix görselleştirme

Çalıştırma:
1. turkiye_2026_bankacilik_verisi.csv dosyasını bu Python dosyasıyla aynı
   klasöre koyun.
2. Gerekli paketleri yükleyin:
   pip install pandas numpy scikit-learn matplotlib seaborn
3. Terminalde çalıştırın:
   python bankacilik_siniflandirma.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# ---------------------------------------------------------
# 1-2. VERİ SETİNİ OKUMA
# ---------------------------------------------------------

DOSYA = "turkiye_2026_bankacilik_verisi.csv"

if not os.path.exists(DOSYA):
    raise FileNotFoundError(
        f"'{DOSYA}' bulunamadı. CSV dosyasını Python dosyasıyla aynı klasöre koyun."
    )

df = pd.read_csv(DOSYA)

print("=" * 70)
print("TÜRKİYE 2026 BANKACILIK VERİSİ - SINIFLANDIRMA PROJESİ")
print("=" * 70)

# ---------------------------------------------------------
# 3. İLK İNCELEME
# ---------------------------------------------------------

print("\n[3] VERİ SETİNİN İLK 5 SATIRI")
print(df.head())

print("\nVeri seti boyutu:")
print(f"Satır sayısı : {df.shape[0]}")
print(f"Sütun sayısı : {df.shape[1]}")

HEDEF = "Hesap_Aktif"

print("\nHedef değişken dağılımı:")
print(df[HEDEF].value_counts())
print("\nHedef değişken yüzdeleri:")
print(df[HEDEF].value_counts(normalize=True).mul(100).round(2))

# ---------------------------------------------------------
# 4. EKSİK DEĞER KONTROLÜ
# ---------------------------------------------------------

print("\n[4] EKSİK DEĞER KONTROLÜ")
eksik = df.isnull().sum()
print(eksik)

if eksik.sum() == 0:
    print("Eksik değer bulunmamaktadır.")
else:
    print("Eksik değerler model ön işleme aşamasında uygun yöntemlerle doldurulacaktır.")

# ---------------------------------------------------------
# 5-6-7. ÖN İŞLEME VE ÖZNİTELİK ÜRETİMİ
# ---------------------------------------------------------

# Benzersiz müşteri ID'si ve müşteri adı doğrudan tahmin için anlamlı
# olmadığı için modelden çıkarılır.
df = df.drop(columns=["Musteri_ID", "Musteri_Adi"], errors="ignore")

# En az 1 yeni öznitelik:
# Toplam borcun kredi kartı limitine oranını hesaplıyoruz.
df["Borc_Limit_Orani"] = np.where(
    df["Kredi_Karti_Limiti_TL"] > 0,
    df["Toplam_Borc_TL"] / df["Kredi_Karti_Limiti_TL"],
    0
)

print("\n[7] ÜRETİLEN YENİ ÖZNİTELİK")
print("Borc_Limit_Orani = Toplam_Borc_TL / Kredi_Karti_Limiti_TL")
print(df["Borc_Limit_Orani"].describe())

# Boolean hedefi 0/1 formatına dönüştür.
y = df[HEDEF].astype(int)
X = df.drop(columns=[HEDEF])

# Kategorik ve sayısal sütunları belirle.
kategorik_sutunlar = X.select_dtypes(include=["object"]).columns.tolist()
sayisal_sutunlar = X.select_dtypes(include=[np.number]).columns.tolist()

print("\nKategorik değişkenler:")
print(kategorik_sutunlar)

print("\nSayısal değişkenler:")
print(sayisal_sutunlar)

# Sayısal değişkenler:
# - Eksik değer varsa median ile doldurulur.
# - StandardScaler ile ölçeklenir.
sayisal_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

# Kategorik değişkenler:
# - Eksik değer varsa en sık değer ile doldurulur.
# - One-Hot Encoding uygulanır.
kategorik_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", sayisal_pipeline, sayisal_sutunlar),
        ("cat", kategorik_pipeline, kategorik_sutunlar),
    ]
)

# ---------------------------------------------------------
# 8. TRAIN / VALIDATION / TEST AYRIMI
# ---------------------------------------------------------

# Önce %80 train+validation ve %20 test.
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Sonra train+validation'ın %25'i validation:
# Sonuç yaklaşık %60 train, %20 validation, %20 test.
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.25,
    random_state=42,
    stratify=y_train_val,
)

print("\n[8] VERİ BÖLÜMLERİ")
print(f"Train      : {len(X_train)}")
print(f"Validation : {len(X_val)}")
print(f"Test       : {len(X_test)}")

# ---------------------------------------------------------
# 9. EN AZ 2 MODEL EĞİTME
# ---------------------------------------------------------

logistic_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                random_state=42
            ),
        ),
    ]
)

knn_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", KNeighborsClassifier(n_neighbors=7)),
    ]
)

print("\n[9] MODELLER EĞİTİLİYOR...")

logistic_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)

# ---------------------------------------------------------
# 10. VALIDATION SONUÇLARI
# ---------------------------------------------------------

def validation_skorlari(model, X_val, y_val):
    tahmin = model.predict(X_val)
    return {
        "Accuracy": accuracy_score(y_val, tahmin),
        "Precision": precision_score(y_val, tahmin, zero_division=0),
        "Recall": recall_score(y_val, tahmin, zero_division=0),
        "F1": f1_score(y_val, tahmin, zero_division=0),
    }


logistic_val = validation_skorlari(logistic_model, X_val, y_val)
knn_val = validation_skorlari(knn_model, X_val, y_val)

validation_sonuclari = pd.DataFrame(
    [logistic_val, knn_val],
    index=["Logistic Regression", "KNN"]
)

print("\n[10] VALIDATION SONUÇLARI")
print(validation_sonuclari.round(4))

# F1-score'a göre modeli seçiyoruz.
if logistic_val["F1"] >= knn_val["F1"]:
    secilen_model_adi = "Logistic Regression"
    secilen_model = logistic_model
else:
    secilen_model_adi = "KNN"
    secilen_model = knn_model

print(f"\nValidation sonucuna göre seçilen model: {secilen_model_adi}")

# ---------------------------------------------------------
# 10-11. TEST DEĞERLENDİRMESİ
# ---------------------------------------------------------

y_test_tahmin = secilen_model.predict(X_test)

accuracy = accuracy_score(y_test, y_test_tahmin)
precision = precision_score(y_test, y_test_tahmin, zero_division=0)
recall = recall_score(y_test, y_test_tahmin, zero_division=0)
f1 = f1_score(y_test, y_test_tahmin, zero_division=0)

cm = confusion_matrix(y_test, y_test_tahmin)

print("\n[11] TEST SONUÇLARI")
print("-" * 50)
print(f"Seçilen Model : {secilen_model_adi}")
print(f"Accuracy      : {accuracy:.4f}")
print(f"Precision     : {precision:.4f}")
print(f"Recall        : {recall:.4f}")
print(f"F1-Score      : {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)

# Confusion Matrix görselleştirme
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Pasif (0)", "Aktif (1)"],
    yticklabels=["Pasif (0)", "Aktif (1)"],
)
plt.title(f"Confusion Matrix - {secilen_model_adi}")
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 12. KISA YORUM
# ---------------------------------------------------------

print("\n[12] KISA YORUM")
print("=" * 70)

if logistic_val["F1"] > knn_val["F1"]:
    print("Validation sonucunda Logistic Regression daha başarılı olmuştur.")
    print(
        "Bunun nedeni, veri setindeki sayısal ve One-Hot Encoding sonrası "
        "oluşan değişkenler arasındaki ilişkilerin daha basit ve doğrusal "
        "bir yapıyla modellenebilmesi olabilir."
    )
elif knn_val["F1"] > logistic_val["F1"]:
    print("Validation sonucunda KNN daha başarılı olmuştur.")
    print(
        "Bunun nedeni, benzer müşteri profillerinin birbirine yakın "
        "olması ve KNN'nin komşuluk ilişkilerinden yararlanabilmesi olabilir."
    )
else:
    print("İki model validation aşamasında benzer F1-score üretmiştir.")
    print(
        "Bu durumda modellerin performansları birbirine yakın olduğundan "
        "test sonucu ve çalışma süresi gibi ölçütler de dikkate alınabilir."
    )

print(f"\nSeçilen modelin test F1-score'u: {f1:.4f}")
print(f"Seçilen modelin test accuracy değeri: {accuracy:.4f}")
print("=" * 70)
