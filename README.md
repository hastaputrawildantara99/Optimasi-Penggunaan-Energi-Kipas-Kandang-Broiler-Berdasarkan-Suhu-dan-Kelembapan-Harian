# Optimasi Penggunaan Energi Kipas Kandang Broiler

Repository ini berisi bahan eksperimen komputasional untuk makalah:

**Analisis Perbandingan Dynamic Programming, Greedy-DFS-BFS, dan Fuzzy Logic dalam Optimasi Penggunaan Energi Kipas Kandang Broiler Berdasarkan Suhu dan Kelembapan Harian**

## Isi Repository

- `src/fan_energy_optimization.py`: kode Python utama untuk mengambil data, menghitung THI, menjalankan algoritma, dan menghasilkan hasil eksperimen. Kode dibuat sederhana dengan `list`, `dict`, `csv`, dan struktur DP/BFS/DFS yang sesuai materi ASA.
- `data/`: data NASA POWER yang digunakan sebagai fallback terbuka dan reprodusibel.
- `results/`: ringkasan metrik dan detail hasil setiap metode.
- `figures/`: grafik perbandingan energi dan THI.
- `sources/`: catatan dataset dan referensi utama.

## Metode

Eksperimen membandingkan:

1. Dynamic Programming
2. Greedy, BFS berlapis, dan DFS bermemoisasi
3. Fuzzy Logic

Objektif optimasi menggabungkan energi relatif kipas, penalti kenyamanan berbasis Temperature-Humidity Index (THI), dan biaya pergantian level kipas.

## Cara Menjalankan

```bash
pip install -r requirements.txt
python src/fan_energy_optimization.py
```

Keluaran utama ada pada folder `data/`, `results/`, dan `figures/`.

## Sumber Data

Dataset kandang unggas Mendeley Data DOI `10.17632/7fptd3rfzy.1` digunakan sebagai kandidat utama. Karena endpoint file publik membutuhkan otorisasi pada lingkungan eksperimen, analisis menggunakan fallback NASA POWER Daily API untuk Semarang.
