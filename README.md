# Animal Habitat Prediction with K-Means Clustering
Proyek ini menggunakan algoritma K-Means Clustering untuk mengelompokkan data hewan berdasarkan tinggi dan warna. Setelah proses pelatihan, pengguna dapat memberikan input tinggi dan warna untuk memprediksi habitat hewan tersebut.

# Dataset
Dataset berupa file CSV bernama Animal Dataset.csv yang berisi kolom berikut:
1. Height (cm) — Tinggi hewan dalam bentuk angka atau rentang seperti 50-70 atau up to 60
2. Color — Warna hewan (contoh: "Grey", "Brown", dsb.)
3. Habitat — Daftar habitat, bisa lebih dari satu (contoh: "Savannas, Grasslands")

# Cara Menjalankan
1. Jalankan skrip: python Kds.py
2. Masukan Animal Dataset.csv pada folder yang sama
3. Pada saat menjalankan program akan menampilkan hasil clustering
4. tutup hasil clustering untuk mulai measukan input
5. input pertama berupa panjang dari binatang
6. input kedua berupa warna dari hewan
7. output berupa habitat terdekat
