# Laporan Proyek Machine Learning - Chatarina Evangelista Sitorus

## Domain Proyek

Kanker paru-paru adalah penyakit yang memiliki tingkat mortalitas tinggi di seluruh dunia, terutama karena seringnya diagnosis baru ditemukan pada stadium lanjut. Gejala awal yang tidak spesifik serta ketergantungan pada metode deteksi invasif menjadi hambatan utama dalam skrining dan penanganan dini kanker paru.

Seiring dengan meningkatnya akses terhadap data kesehatan dan berkembangnya teknologi kecerdasan buatan, pendekatan berbasis machine learning (ML) mulai banyak digunakan dalam analisis risiko penyakit kronis, termasuk kanker paru-paru. ML menawarkan pendekatan non-invasif dan cepat dengan menganalisis data sederhana seperti usia, kebiasaan merokok, dan gejala awal.

Studi oleh **Chaturvedi et al. (2021)** menunjukkan bahwa berbagai teknik machine learning mampu memprediksi dan mengklasifikasikan kanker paru secara efektif menggunakan data gejala dan riwayat pasien. Penelitian lain oleh **Patra (2020)** juga membuktikan bahwa algoritma ML seperti Decision Tree dan Support Vector Machine dapat menghasilkan prediksi yang akurat hanya dengan input klinis dasar.

Sementara itu, **Gould et al. (2022)** menekankan bahwa data klinis rutin seperti riwayat pemeriksaan dan laboratorium dapat diintegrasikan dengan model pembelajaran mesin untuk meningkatkan akurasi identifikasi dini kanker paru-paru secara praktis di dunia nyata.

### Referensi

- Chaturvedi, P., Jhamb, A., Vanani, M., & Nemade, V. (2021). Prediction and classification of lung cancer using machine learning techniques. *IOP Conference Series: Materials Science and Engineering, 1099*(1), 012059. https://doi.org/10.1088/1757-899X/1099/1/012059  
- Patra, R. (2020). Prediction of lung cancer using machine learning classifier. In S. C. Satapathy, V. Bhateja, & S. Das (Eds.), *Data management, analytics and innovation* (pp. 107–120). Springer. https://doi.org/10.1007/978-981-15-6648-6_11  
- Gould, M. K., Huang, B. Z., Tammemagi, M. C., Kinar, Y., & Shiff, R. (2022). Machine learning for early lung cancer identification using routine clinical and laboratory data. *American Journal of Respiratory and Critical Care Medicine, 205*(3), 350–352. https://doi.org/10.1164/rccm.202007-2791OC

## Business Understanding

Dalam upaya mendukung deteksi dini kanker paru-paru, proyek ini berfokus pada pengembangan sistem klasifikasi berbasis machine learning (ML) yang menggunakan data klinis non-invasif sebagai sumber utama informasi. ML dipilih karena kemampuannya dalam mengenali pola tersembunyi dari data sederhana seperti usia, kebiasaan merokok, dan gejala awal pasien, tanpa perlu bergantung pada metode diagnostik invasif atau mahal.

Berdasarkan latar belakang tersebut, maka proses klarifikasi masalah dan tujuan yang ingin dicapai dalam studi ini dirumuskan sebagai berikut:

### Problem Statements

1. Kanker paru-paru sering kali terdiagnosis pada stadium lanjut karena gejala awalnya tidak spesifik dan sulit dikenali.
2. Deteksi awal memerlukan pemeriksaan mahal dan invasif seperti CT-scan dan biopsi yang belum dapat diakses semua fasilitas kesehatan.
3. Belum tersedia sistem berbasis data klinis sederhana yang mampu mengidentifikasi risiko kanker paru secara cepat dan efektif.
4. Diperlukan pendekatan prediktif menggunakan machine learning untuk membangun model klasifikasi risiko kanker paru yang dapat diandalkan.

### Goals

Dengan memanfaatkan machine learning, studi ini bertujuan untuk:

1. Membangun model klasifikasi yang mampu memprediksi status kanker paru-paru berdasarkan fitur-fitur klinis dasar.
2. Membandingkan performa dua algoritma yaitu Logistic Regression dan Random Forest Classifier sebagai alternatif solusi.
3. Memberikan dasar teknis dalam pengembangan sistem skrining awal kanker paru-paru berbasis data terbuka yang hemat biaya dan informatif.

### Solution statements

- **Model 1: Logistic Regression**  
  Dipilih sebagai baseline model karena sifatnya yang cepat, sederhana, serta mudah diinterpretasikan. Cocok sebagai titik awal untuk evaluasi klasifikasi biner.

- **Model 2: Random Forest Classifier**  
  Merupakan model ansambel berbasis decision tree yang lebih kompleks dan mampu menangkap hubungan non-linear. Model ini juga relatif tahan terhadap overfitting.

Evaluasi kinerja model dilakukan dengan menggunakan metrik klasifikasi yang relevan: **Accuracy**, **Precision**, **Recall**, dan **F1-Score**. Visualisasi seperti confusion matrix dan grafik perbandingan juga digunakan untuk memberikan insight tambahan dalam pemilihan model terbaik.


## Data Understanding

Dataset yang digunakan dalam proyek ini berjudul **Lung Cancer Dataset** dan tersedia secara publik melalui platform Kaggle ([Lung Cancer Dataset](https://www.kaggle.com/datasets/akashnath29/lung-cancer-dataset)). Dataset ini terdiri dari **3000 entri pasien** dengan **15 fitur input** dan **1 kolom target** (`LUNG_CANCER`). Data bersifat **non-citra**, yaitu berdasarkan survei atau wawancara medis, sehingga cocok untuk diterapkan dalam sistem skrining awal berbasis data klinis.

### Struktur Dataset

Dataset dimuat menggunakan `pandas.read_csv()` dan memiliki struktur sebagai berikut:

```python
df_lung.shape
```

Output:
```
(3000, 16)
```

Artinya terdapat **3000 baris** (pasien) dan **16 kolom**, termasuk variabel target.

### Deskripsi Fitur

| Fitur | Deskripsi |
|-------|-----------|
| `GENDER` | Jenis kelamin pasien (`M` = Laki-laki, `F` = Perempuan) |
| `AGE` | Usia pasien (numerik) |
| `SMOKING` | Riwayat merokok (`1` = Tidak, `2` = Ya) |
| `YELLOW_FINGERS` | Ada/tidaknya jari menguning akibat nikotin |
| `ANXIETY` | Gangguan kecemasan |
| `PEER_PRESSURE` | Terpengaruh tekanan sosial |
| `CHRONIC_DISEASE` | Riwayat penyakit kronis |
| `FATIGUE` | Kelelahan kronis |
| `ALLERGY` | Riwayat alergi |
| `WHEEZING` | Napas berbunyi |
| `ALCOHOL_CONSUMING` | Konsumsi alkohol |
| `COUGHING` | Batuk kronis |
| `SHORTNESS_OF_BREATH` | Sesak napas |
| `SWALLOWING_DIFFICULTY` | Sulit menelan |
| `CHEST_PAIN` | Nyeri dada |
| `LUNG_CANCER` | Target klasifikasi (`YES` = Kanker, `NO` = Tidak) |

### Eksplorasi Data

Beberapa tahapan awal eksplorasi data dilakukan untuk memahami karakteristik dataset:

#### 1. Pemeriksaan Unik dan Duplikasi

```python
df_lung.nunique()
df_lung.duplicated().sum()
```

Hasil menunjukkan tidak ada nilai kosong dan hanya terdapat **2 data duplikat** yang dapat dihapus.

#### 2. Distribusi Data

##### Visualisasi Korelasi Awal

```python
sns.heatmap(df_lung.corr(numeric_only=True), annot=True)
```

![Heatmap Awal](assets/eda/heatmap.png)

##### Deteksi Outlier dengan Boxplot (AGE)

```python
sns.boxplot(x=df_lung["AGE"])
```

![Boxplot AGE](assets/eda/boxplot.png)

##### Distribusi Gender dan Label

```python
sns.countplot(x="GENDER", data=df_lung)
sns.countplot(x="LUNG_CANCER", data=df_lung)
```

![Distribusi Gender dan Status Kanker](assets/eda/countplot.png)


##### Distribusi Usia Pasien

```python
sns.histplot(data=df_lung, x="AGE", kde=True)
```

![Distribusi Usia](assets/eda/histplot.png)

---

Paragraf awal bagian ini menjelaskan informasi mengenai data yang Anda gunakan dalam proyek. Sertakan juga sumber atau tautan untuk mengunduh dataset. Contoh: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Restaurant+%26+consumer+data).

Selanjutnya uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

### Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
- accepts : merupakan jenis pembayaran yang diterima pada restoran tertentu.
- cuisine : merupakan jenis masakan yang disajikan pada restoran.
- dst

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis.

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

