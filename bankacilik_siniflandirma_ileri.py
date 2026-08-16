"""
ÖDEV: Türkiye 2026 Bankacılık Verisi ile Hesap Aktiflik Tahmini

Amaç:
Bu projede banka müşterilerinin hesaplarının aktif olup olmadığını tahmin
eden bir makine öğrenmesi sınıflandırma modeli geliştirilmiştir. Veri seti
incelenmiş, eksik ve aykırı değerler kontrol edilmiş, kategorik değişkenler
One-Hot Encoding ile dönüştürülmüş, yeni öznitelikler üretilmiş ve özellik
seçimi uygulanmıştır. Logistic Regression, KNN, Decision Tree ve Random
Forest modelleri karşılaştırılmıştır. En iyi model Grid Search ile
iyileştirilmiş ve test verisi üzerinde değerlendirilmiştir.

Problem türü:
İkili sınıflandırma (Binary Classification).

Hedef değişken:
Hesap_Aktif
0 = Pasif hesap
1 = Aktif hesap

Kullanılan kütüphaneler:
- pandas, numpy: veri okuma ve veri işleme
- scikit-learn: ön işleme, modelleme, özellik seçimi, Grid Search ve metrikler
- matplotlib, seaborn: grafikler ve confusion matrix

Çalıştırma:
1. turkiye_2026_bankacilik_verisi.csv dosyasını bu Python dosyasıyla aynı
   klasöre koyun.
2. Gerekli paketleri yükleyin:
   pip install pandas numpy scikit-learn matplotlib seaborn
3. Çalıştırın:
   python bankacilik_siniflandirma_ileri.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

warnings.filterwarnings("ignore")

# =========================================================
# 1-2. VERİ SETİNİ OKUMA VE PROBLEM AÇIKLAMASI
# =========================================================

DOSYA = "turkiye_2026_bankacilik_verisi.csv"
HEDEF = "Hesap_Aktif"

if not os.path.exists(DOSYA):
    raise FileNotFoundError(
        f"'{DOSYA}' bulunamadı. CSV dosyasını Python dosyasıyla aynı klasöre koyun."
    )

df = pd.read_csv(DOSYA)

print("=" * 80)
print("TÜRKİYE 2026 BANKACILIK VERİSİ - MAKİNE ÖĞRENMESİ PROJESİ")
print("=" * 80)

print("""
VERİ SETİ AÇIKLAMASI:
Veri seti banka müşterilerine ait yaş, şehir, meslek, gelir tipi,
eğitim durumu, kart tipi, kredi kartı limiti, toplam borç,
limit kullanım oranı, ödeme bilgileri, gecikme bilgileri ve müşteri
skoru gibi bilgileri içermektedir.

ÇÖZÜLEN PROBLEM:
Amaç, müşterinin banka hesabının aktif olup olmadığını tahmin etmektir.

PROBLEM TÜRÜ:
İkili sınıflandırma (Binary Classification).

HEDEF DEĞİŞKEN:
Hesap_Aktif
0 -> Pasif
1 -> Aktif
""")

# =========================================================
# 3-4. HEDEF DEĞİŞKEN VE TEMEL VERİ İNCELEMESİ
# =========================================================

print("\n[3] HEDEF DEĞİŞKEN")
print("Hedef değişken:", HEDEF)
print("Problem türü: İkili Sınıflandırma")

print("\nHedef değişken dağılımı:")
print(df[HEDEF].value_counts())

print("\nHedef değişken yüzdeleri:")
print(df[HEDEF].value_counts(normalize=True).mul(100).round(2))

print("\n[4] İLK 10 SATIR")
print(df.head(10))

print("\nSATIR-SÜTUN SAYISI")
print("Satır sayısı :", df.shape[0])
print("Sütun sayısı :", df.shape[1])

print("\nVERİ TİPLERİ")
print(df.dtypes)

print("\nTEMEL İSTATİSTİKLER")
print(df.describe(include="all").T)

# =========================================================
# 5. EKSİK DEĞER KONTROLÜ
# =========================================================

print("\n[5] EKSİK DEĞER KONTROLÜ")
eksik = df.isnull().sum()
eksik_yuzde = (df.isnull().mean() * 100).round(2)

eksik_tablo = pd.DataFrame({
    "Eksik_Sayi": eksik,
    "Eksik_Yuzde": eksik_yuzde
})

print(eksik_tablo)

if eksik.sum() == 0:
    print("\nEksik değer bulunmamaktadır.")
else:
    print("\nEksik değerler model pipeline içinde median/mode ile doldurulacaktır.")

# =========================================================
# 6-7. VERİ HAZIRLAMA VE AYKIRI DEĞER İNCELEMESİ
# =========================================================

# Benzersiz ID ve isim tahmin açısından anlamlı olmadığı için çıkarılıyor.
df = df.drop(columns=["Musteri_ID", "Musteri_Adi"], errors="ignore")

# Boolean hedefi 0/1 yap.
df[HEDEF] = df[HEDEF].astype(int)

# ---------------------------------------------------------
# 7. AYKIRI DEĞER ANALİZİ
# ---------------------------------------------------------

print("\n[7] AYKIRI DEĞER ANALİZİ")

numeric_for_outlier = df.drop(columns=[HEDEF]).select_dtypes(
    include=np.number
).columns

outlier_rows = []

for col in numeric_for_outlier:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1

    if iqr == 0:
        outlier_count = 0
    else:
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_count = ((df[col] < lower) | (df[col] > upper)).sum()

    outlier_rows.append({
        "Degisken": col,
        "Aykiri_Deger_Sayisi": int(outlier_count)
    })

outlier_df = pd.DataFrame(outlier_rows)
print(outlier_df.sort_values(
    "Aykiri_Deger_Sayisi",
    ascending=False
).head(15))

print("""
AYKIRI DEĞER YAKLAŞIMI:
Aykırı değerler veri setinden doğrudan silinmemiştir. Bankacılık verilerinde
yüksek borç, yüksek limit veya ödeme değerleri gerçek müşterileri temsil
edebileceği için bu değerlerin silinmesi bilgi kaybına yol açabilir.
Sayısal değişkenler model pipeline aşamasında ölçeklenmekte ve model
performansı validation/test sonuçları üzerinden değerlendirilmektedir.
""")

# =========================================================
# 8-9. ÖZNİTELİK MÜHENDİSLİĞİ
# =========================================================

# 1. Borç / kredi kartı limiti oranı
df["Borc_Limit_Orani"] = np.where(
    df["Kredi_Karti_Limiti_TL"] > 0,
    df["Toplam_Borc_TL"] / df["Kredi_Karti_Limiti_TL"],
    0
)

# 2. Ödeme / borç oranı
df["Odeme_Borc_Orani"] = np.where(
    df["Toplam_Borc_TL"] > 0,
    df["Son_Ay_Odenen_TL"] / df["Toplam_Borc_TL"],
    0
)

# 3. Gecikme riski
df["Gecikme_Var_Mi"] = (df["Gecikme_Gun_Sayisi"] > 0).astype(int)

print("\n[9] ÜRETİLEN ÖZNİTELİKLER")
print("1. Borc_Limit_Orani = Toplam_Borc_TL / Kredi_Karti_Limiti_TL")
print("2. Odeme_Borc_Orani = Son_Ay_Odenen_TL / Toplam_Borc_TL")
print("3. Gecikme_Var_Mi = Gecikme_Gun_Sayisi > 0")

print("\nYeni özniteliklerin temel istatistikleri:")
print(
    df[
        [
            "Borc_Limit_Orani",
            "Odeme_Borc_Orani",
            "Gecikme_Var_Mi"
        ]
    ].describe()
)

# =========================================================
# 10. ÖZNİTELİK SEÇİMİ
# =========================================================

print("\n[10] ÖZNİTELİK SEÇİMİ")
print("""
Mutual Information (Karşılıklı Bilgi) yöntemi kullanılacaktır.
Bu yöntem, değişken ile hedef değişken arasındaki bilgi ilişkisini
ölçerek en anlamlı değişkenleri seçmek için kullanılır.

Özellik seçimi veri sızıntısını önlemek amacıyla Pipeline içinde
SelectKBest ile uygulanmaktadır.
""")

# =========================================================
# 11. TRAIN / VALIDATION / TEST
# =========================================================

X = df.drop(columns=[HEDEF])
y = df[HEDEF]

# %20 test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Kalan %80'in %25'i validation = toplam %20
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.25,
    random_state=42,
    stratify=y_train_val
)

print("\n[11] VERİ BÖLÜMLERİ")
print(f"Train      : {len(X_train)} (%60)")
print(f"Validation : {len(X_val)} (%20)")
print(f"Test       : {len(X_test)} (%20)")

# =========================================================
# ÖN İŞLEME
# =========================================================

categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

numeric_columns = X.select_dtypes(
    include=[np.number]
).columns.tolist()

print("\nKategorik değişken sayısı:", len(categorical_columns))
print("Sayısal değişken sayısı:", len(numeric_columns))

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_columns),
        ("cat", categorical_pipeline, categorical_columns),
    ]
)

# =========================================================
# 12. EN AZ 3 MODEL EĞİTME
# =========================================================

# Özellik seçimi:
# 50 adet özellik seçilir. Eğer veri daha az özellik üretirse min değeri kullanılır.
# SelectKBest Pipeline içerisinde ön işlemeden sonra uygulanır.
feature_selector = SelectKBest(
    score_func=mutual_info_classif,
    k=50
)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=42
    ),
    "KNN": KNeighborsClassifier(
        n_neighbors=7
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=10,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    ),
}

trained_models = {}
validation_results = []

print("\n[12] MODELLER EĞİTİLİYOR...")

for name, model in models.items():

    # SelectKBest k=50 kullanıldığı için veri dönüşümünden sonra
    # 50'den az özellik oluşması durumuna karşı dinamik pipeline oluşturuyoruz.
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("feature_selection", SelectKBest(
                score_func=mutual_info_classif,
                k=50
            )),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)
    trained_models[name] = pipeline

    val_prediction = pipeline.predict(X_val)

    acc = accuracy_score(y_val, val_prediction)
    precision = precision_score(
        y_val, val_prediction, zero_division=0
    )
    recall = recall_score(
        y_val, val_prediction, zero_division=0
    )
    f1 = f1_score(
        y_val, val_prediction, zero_division=0
    )

    validation_results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    })

validation_df = pd.DataFrame(validation_results)
validation_df = validation_df.sort_values(
    "F1-Score",
    ascending=False
)

print("\n[13] VALIDATION MODEL KARŞILAŞTIRMASI")
print(validation_df.round(4).to_string(index=False))

best_model_name = validation_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

print(f"\nValidation sonucuna göre en iyi model: {best_model_name}")

# =========================================================
# 14. GRID SEARCH
# =========================================================

print("\n[14] GRID SEARCH")
print("En iyi model için basit hiperparametre ayarlaması yapılıyor...")

if best_model_name == "Logistic Regression":

    param_grid = {
        "feature_selection__k": [30, 50],
        "model__C": [0.1, 1],
        "model__solver": ["liblinear"],
    }

elif best_model_name == "KNN":

    param_grid = {
        "feature_selection__k": [30, 50],
        "model__n_neighbors": [5, 7, 9],
        "model__weights": ["uniform", "distance"],
    }

elif best_model_name == "Decision Tree":

    param_grid = {
        "feature_selection__k": [30, 50],
        "model__max_depth": [10, 15],
        "model__min_samples_split": [2, 5],
    }

else:  # Random Forest

    param_grid = {
        "feature_selection__k": [30, 50],
        "model__n_estimators": [100],
        "model__max_depth": [8, 12],
        "model__min_samples_split": [2, 5],
    }

grid_search = GridSearchCV(
    estimator=best_model,
    param_grid=param_grid,
    scoring="f1",
    cv=2,
    n_jobs=2,
    verbose=0
)

grid_search.fit(X_train, y_train)

tuned_model = grid_search.best_estimator_

print("\nGrid Search en iyi parametreleri:")
print(grid_search.best_params_)

print(f"Grid Search en iyi CV F1: {grid_search.best_score_:.4f}")

val_tuned_prediction = tuned_model.predict(X_val)
val_tuned_f1 = f1_score(
    y_val,
    val_tuned_prediction,
    zero_division=0
)

print(f"Grid Search sonrası validation F1: {val_tuned_f1:.4f}")

# =========================================================
# 15. TEST DEĞERLENDİRMESİ
# =========================================================

print("\n[15] TEST DEĞERLENDİRMESİ")

y_test_prediction = tuned_model.predict(X_test)

test_accuracy = accuracy_score(
    y_test,
    y_test_prediction
)

test_precision = precision_score(
    y_test,
    y_test_prediction,
    zero_division=0
)

test_recall = recall_score(
    y_test,
    y_test_prediction,
    zero_division=0
)

test_f1 = f1_score(
    y_test,
    y_test_prediction,
    zero_division=0
)

cm = confusion_matrix(
    y_test,
    y_test_prediction
)

print("-" * 60)
print("Seçilen Model:", best_model_name)
print(f"Accuracy  : {test_accuracy:.4f}")
print(f"Precision : {test_precision:.4f}")
print(f"Recall    : {test_recall:.4f}")
print(f"F1-Score  : {test_f1:.4f}")

print("\nConfusion Matrix:")
print(cm)

# Confusion Matrix grafiği
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Pasif (0)", "Aktif (1)"],
    yticklabels=["Pasif (0)", "Aktif (1)"]
)
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.tight_layout()
plt.show()

# =========================================================
# 16-17. MODEL YORUMU VE AÇIKLANABİLİRLİK
# =========================================================

print("\n[16-17] MODEL YORUMU")
print("=" * 80)

print(f"""
Validation sonuçlarına göre en başarılı model: {best_model_name}

Model seçimi F1-score üzerinden yapılmıştır. F1-score; precision ve
recall değerlerini birlikte değerlendirdiği için banka müşterilerinin
aktif/pasif sınıflandırmasında dengeli bir performans ölçüsü olarak
kullanılmıştır.

Test sonuçları:
- Accuracy  : {test_accuracy:.4f}
- Precision : {test_precision:.4f}
- Recall    : {test_recall:.4f}
- F1-Score  : {test_f1:.4f}

MODEL SINIRLILIKLARI:
1. Veri setindeki değişkenler belirli bir dönem/senaryoyu temsil ediyor olabilir.
2. Modelin gerçek bankacılık ortamındaki başarısı farklı veri üzerinde değişebilir.
3. One-Hot Encoding sonucunda çok sayıda değişken oluşması model karmaşıklığını artırabilir.
4. Hesap aktifliği sadece bu veri setindeki değişkenlerle açıklanamayabilir.
""")

# ---------------------------------------------------------
# BONUS: Mutual Information skorlarının yorumlanması
# ---------------------------------------------------------

print("\nBONUS - ÖZNİTELİK SEÇİMİ / ÖNEMLİ DEĞİŞKENLER")

try:
    fitted_preprocessor = tuned_model.named_steps["preprocessor"]
    selector = tuned_model.named_steps["feature_selection"]

    feature_names = fitted_preprocessor.get_feature_names_out()
    selected_mask = selector.get_support()

    selected_features = feature_names[selected_mask]
    selected_scores = selector.scores_[selected_mask]

    importance_df = pd.DataFrame({
        "Feature": selected_features,
        "Mutual_Information": selected_scores
    }).sort_values(
        "Mutual_Information",
        ascending=False
    )

    print("\nMutual Information açısından en önemli 15 özellik:")
    print(importance_df.head(15).to_string(index=False))

except Exception as e:
    print("Öznitelik önemleri alınırken hata oluştu:", e)

print("\n" + "=" * 80)
print("PROJE TAMAMLANDI.")
print("=" * 80)
