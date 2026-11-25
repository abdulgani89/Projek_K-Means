# K-Means Clustering Web App (Django)

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)
![Django](https://img.shields.io/badge/Django-Web_Framework-092E20?style=flat&logo=django&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Algorithm-K--Means_Clustering-FF6F00?style=flat)
![License](https://img.shields.io/badge/License-MIT-blue?style=flat)

Aplikasi web berbasis **Django** untuk mengelompokkan data secara otomatis menggunakan algoritma **K-Means Clustering** dan menampilkan hasil segmentasi dalam bentuk tabel serta visualisasi interaktif.

---

## Preview Aplikasi

| Halaman Input & Parameter `K` | Hasil Segmentasi & Visualisasi Klaster |
| :---: | :---: |
| ![Input Data](docs/preview-input.png) | ![Hasil Klaster](docs/preview-result.png) |

> *Ganti gambar di folder `docs/` dengan tangkapan layar aplikasi Anda.*

---

## Latar Belakang & Tujuan

Analisis pengelompokan data sering kali hanya berhenti di notebook dan sulit digunakan oleh pengguna non-teknis. Proyek ini membungkus pipeline **K-Means Clustering** ke dalam antarmuka web agar pengguna dapat:
- Mengolah dataset **[sebutkan objek data, mis. data siswa / pelanggan / penjualan]** tanpa menulis kode.
- Menentukan jumlah klaster (`K`) secara fleksibel sesuai kebutuhan analisis.
- Membaca hasil segmentasi dan profil tiap klaster secara langsung di browser.

---

## Fitur Utama

- **Manajemen & Input Dataset:** Unggah atau kelola data yang akan dianalisis langsung melalui antarmuka web.
- **Konfigurasi Klaster Dinamis:** Pengguna dapat menentukan jumlah klaster (`K`) sebelum proses komputasi dijalankan.
- **Komputasi K-Means Otomatis:** Melakukan normalisasi data, perhitungan jarak Euclidean, pembaruan centroid, hingga konvergen.
- **Ringkasan & Visualisasi Hasil:** Menampilkan anggota tiap klaster, posisi centroid, serta ringkasan statistik per kelompok.

---

## Alur Kerja Sistem

1. **Data Input:** Pengguna memasukkan atau mengunggah dataset ke sistem.
2. **Preprocessing:** Pembersihan dan penyesuaian skala fitur numerik agar jarak antar variabel seimbang.
3. **Clustering (K-Means):** Algoritma membagi data ke dalam `K` klaster berdasarkan kedekatan jarak terhadap centroid.
4. **Output & Visualisasi:** Hasil pengelompokan disimpan dan disajikan dalam tampilan web yang mudah dibaca.

---

## Tech Stack

| Kategori | Teknologi |
| :--- | :--- |
| **Backend** | Python, Django |
| **Machine Learning & Data** | `[isi sesuai requirements.txt, mis. Scikit-learn, Pandas, NumPy]` |
| **Visualisasi** | `[mis. Matplotlib / Chart.js / Plotly]` |
| **Frontend** | HTML5, CSS3, `[mis. Bootstrap / Tailwind CSS]` |
| **Database** | SQLite (Development) |

---

## Struktur Proyek

```text
Projek_K-Means/
├── clustering/           # App utama: views, models, logika K-Means, templates
├── kmeans_project/       # Konfigurasi proyek Django (settings, urls)
├── docs/                 # Aset tangkapan layar untuk dokumentasi
├── .env.example          # Template konfigurasi environment
├── .gitignore
├── manage.py
├── README.md
└── requirements.txt
```

---

## Cara Menjalankan di Lokal

1. **Clone repositori**
   ```powershell
   git clone [https://github.com/abdulgani89/Projek_K-Means.git](https://github.com/abdulgani89/Projek_K-Means.git)
   cd Projek_K-Means
   ```

2. **Buat dan aktifkan virtual environment**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install dependensi**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Siapkan environment variable**
   Salin `.env.example` menjadi `.env`, lalu sesuaikan nilai `SECRET_KEY` dan `DEBUG`.

5. **Jalankan migrasi dan server lokal**
   ```powershell
   python manage.py migrate
   python manage.py runserver
   ```
   Akses aplikasi melalui `http://127.0.0.1:8000/`.

---

## Pengembangan Selanjutnya

- [ ] Penambahan **Elbow Method** dan **Silhouette Score** untuk rekomendasi jumlah `K` optimal.
- [ ] Fitur ekspor hasil klaster ke format `.csv` atau `.xlsx`.
- [ ] Deployment demo online agar dapat dicoba langsung tanpa instalasi lokal.

---

## Author

**Muhamad Abdul Gani**
- GitHub: [@abdulgani89](https://github.com/abdulgani89)
