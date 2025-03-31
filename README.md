# Analisis Sentimen dan Pemrosesan Teks dengan Python menggunakan Ulasan Amazon

## Ringkasan
Proyek ini berfokus pada analisis sentimen menggunakan dataset yang berisi ulasan-ulasan di platform Amazon. Prosesnya mencakup prapemrosesan teks, visualisasi, rekayasa fitur, dan pemodelan topik menggunakan teknik pembelajaran mesin.

## Fitur
- Memuat dan memproses data
- Menangani nilai yang hilang dan data duplikat
- Pembersihan teks (menghapus stopwords, lemmatization, tanda baca, dll.)
- Analisis eksploratif data (EDA) dengan visualisasi
- Ekstraksi fitur menggunakan vektorisasi TF-IDF
- Pemodelan topik dengan Latent Dirichlet Allocation (LDA)
- Klasifikasi sentimen menggunakan regresi logistik

## Persyaratan
Pastikan Anda telah menginstal pustaka berikut sebelum menjalankan skrip:

```bash
pip install numpy pandas matplotlib seaborn wordcloud scipy nltk scikit-learn gensim
```

## Dataset
Skrip ini membaca dataset bernama `Dataset_Pertama_train.csv`. Pastikan dataset tersedia di Google Drive jika menggunakan Google Colab.

## Cara Penggunaan di Google Colab
1. **Buka Google Colab**: Kunjungi [Google Colab](https://colab.research.google.com/drive/1o3tclPbOsLlm5dcz3V7tbq70e6dpV4dF?usp=sharing).
2. **Hubungkan Google Drive**: Jika dataset ada di Google Drive, jalankan kode berikut untuk mengaksesnya:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. **Arahkan ke Direktori Dataset**:
   ```python
   import os
   os.chdir("/content/drive/MyDrive/Path/Ke/Dataset")
   ```
4. **Unggah dan Jalankan Skrip**:
   - Salin kode dari `fesmaro.py` ke dalam sel di notebook.
   - Jalankan setiap sel satu per satu sesuai urutan eksekusi.

## Penjelasan Kode
### 1. Memuat dan Mengeksplorasi Data
- Membaca file CSV yang berisi data ulasan.
- Menampilkan informasi dataset dan nilai yang hilang.
- Memvisualisasikan distribusi kelas menggunakan Seaborn.

### 2. Prapemrosesan Teks
- Mengubah teks menjadi huruf kecil.
- Menghapus angka, tanda baca, dan stopwords.
- Menerapkan lemmatization menggunakan `WordNetLemmatizer` dari `nltk`.
- Membersihkan data yang hilang dan duplikat.

### 3. Visualisasi Data
- Menampilkan word cloud dari kata-kata yang sering muncul.
- Menunjukkan distribusi panjang teks ulasan.
- Memplot kata-kata yang paling sering muncul dalam ulasan menggunakan `matplotlib` dan `seaborn`.

### 4. Rekayasa Fitur
- Menggunakan vektorisasi TF-IDF (`TfidfVectorizer` dari `sklearn`) untuk mentransformasikan data teks menjadi fitur numerik.
- Melakukan pemodelan topik menggunakan Latent Dirichlet Allocation (LDA) dari `gensim`.

### 5. Analisis Sentimen
- Menganalisis proporsi ulasan positif dan negatif menggunakan `value_counts()` dari pandas.
- Memberikan wawasan tentang distribusi sentimen dengan diagram batang.
- Menyoroti kata-kata yang sering muncul dalam ulasan tertentu, seperti kata "shipping."

### 6. Pelatihan Model
- Menggunakan regresi logistik (`LogisticRegression` dari `sklearn`) untuk klasifikasi sentimen.
- Membagi dataset menjadi data latih dan uji menggunakan `train_test_split()`.
- Mengevaluasi akurasi model dengan `classification_report()` dan `confusion_matrix()`.

## Contoh Output
- Grafik distribusi kelas sentimen.
- Visualisasi word cloud untuk kata-kata yang paling sering muncul.
- Histogram panjang teks ulasan.
- Tabel topik hasil pemodelan LDA.
- Akurasi dan metrik evaluasi model klasifikasi.

## Catatan
- Pastikan pustaka `nltk` telah diunduh sebelum menjalankan skrip:
  ```python
  import nltk
  nltk.download('stopwords')
  nltk.download('wordnet')
  ```
- Sesuaikan path dataset jika diperlukan.
- Jika dataset tidak di Google Drive, bisa langsung diunggah ke Google Colab dengan:
  ```python
  from google.colab import files
  uploaded = files.upload()
  ```


# Sentiment Analysis with TF-IDF and Machine Learning

## Deskripsi
Program ini merupakan sistem analisis sentimen berbasis machine learning yang menggunakan TF-IDF sebagai teknik ekstraksi fitur dan model yang telah dilatih sebelumnya untuk melakukan klasifikasi sentimen terhadap teks review.

## Fitur
- Membersihkan teks dengan menghapus angka, tanda baca, dan stopwords
- Melakukan lemmatization untuk normalisasi kata
- Menggunakan model machine learning yang telah dilatih untuk memprediksi sentimen
- Input teks secara interaktif melalui terminal

## Instalasi
### 1. Clone Repository
```bash
git clone https://github.com/AgentlReal/BigDataFesmaro.git
cd BigDataFesmaro
cd "Model Testing Program"
```
### 2. Instal Dependensi
Pastikan Anda memiliki Python terinstal, kemudian jalankan:
```bash
pip install -r requirements.txt
```

### 3. Download NLTK Resources
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

## Penggunaan
1. Pastikan file `tfidf_vectorizer.pkl` dan `best_model.pkl` tersedia di direktori proyek.
2. Jalankan program dengan perintah:
```bash
python test.py
```
3. Masukkan teks review, lalu tekan enter untuk mendapatkan hasil prediksi sentimen.
4. Ketik `exit` untuk keluar dari program.

## Struktur Proyek
```
/Model Testing Program/
|-- test.py  # File utama
|-- tfidf_vectorizer.pkl   # Model TF-IDF yang telah dilatih
|-- best_model.pkl         # Model machine learning yang telah dilatih
|-- requirements.txt       # Daftar dependensi Python
|-- README.md              # Dokumentasi proyek
```

## Dependensi
- Python 3
- scikit-learn
- joblib
- nltk


## Lisensi
Proyek ini bersifat open-source dan tersedia di bawah lisensi MIT.

