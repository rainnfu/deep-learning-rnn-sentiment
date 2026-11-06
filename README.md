# Implementasi Deep Learning: Klasifikasi Sentimen dengan RNN & GRU

Repository ini dibuat untuk memenuhi tugas individu mata kuliah **Deep Learning**. Proyek ini mengimplementasikan dan membandingkan arsitektur **Vanilla RNN (`SimpleRNN`)** dan **Gated Recurrent Unit (`GRU`)** untuk analisis sentimen pada teks ulasan film.

## 📌 Informasi Mahasiswa
- **Nama:** Raihan Ramadhan
- **NIM:** H071241021

## 📊 Dataset
Dataset yang digunakan adalah **IMDB Movie Reviews Dataset** (tersedia secara bawaan di `tensorflow.keras.datasets.imdb`):
- **Jumlah Data:** 50.000 ulasan film (25.000 *train*, 25.000 *test*).
- **Kelas Target:** Biner — Positif (`1`) dan Negatif (`0`).
- **Preprocessing:** Tokenisasi 10.000 kata terpopuler dan *sequence padding* (`maxlen=200`).

## ⚙️ Arsitektur Model
1. **Model 1 — Vanilla RNN (`SimpleRNN`)**
   - `Embedding(10000, 64)` -> `SimpleRNN(64)` -> `Dropout(0.5)` -> `Dense(1, activation='sigmoid')`
2. **Model 2 — Gated Recurrent Unit (`GRU`)**
   - `Embedding(10000, 64)` -> `GRU(64)` -> `Dropout(0.5)` -> `Dense(1, activation='sigmoid')`

Kedua model dilatih menggunakan optimizer `Adam`, *loss function* `binary_crossentropy`, `batch_size=128`, selama 5 *epoch*.

## 🚀 Cara Menjalankan
1. Buka file `RNN_GRU_Sentiment_Analysis.ipynb` menggunakan Google Colab atau Jupyter Notebook.
2. Jalankan seluruh *cell* dari awal hingga akhir (tidak memerlukan unduhan dataset manual).

## 📈 Kesimpulan Singkat
Arsitektur **GRU** mampu mengatasi masalah *vanishing gradient* pada teks panjang (200 kata) jauh lebih baik dibandingkan **SimpleRNN**, sehingga menghasilkan akurasi pengujian (*test accuracy*) yang lebih tinggi dan *loss* yang lebih stabil.
