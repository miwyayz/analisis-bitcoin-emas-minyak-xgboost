# Analisis Korelasi dan Prediksi Harga Bitcoin Berdasarkan Harga Emas dan Minyak Menggunakan XGBoost

## Deskripsi

Penelitian ini menganalisis hubungan dan pola pergerakan harga **Bitcoin, emas, dan minyak** selama periode **2020–2026**. Analisis dilakukan untuk melihat tren harga, hubungan antara Bitcoin dengan emas dan minyak, serta pola perubahan harga emas dan minyak ketika Bitcoin mengalami kenaikan atau penurunan.

Selain analisis korelasi dan pola perubahan, penelitian ini menggunakan **XGBoost** untuk menguji kemampuan data historis Bitcoin, emas, dan minyak dalam melakukan prediksi harga Bitcoin.

## Pertanyaan Penelitian

1. Bagaimana tren perubahan harga Bitcoin, emas, dan minyak selama periode 2020–2026?
2. Seberapa kuat hubungan antara harga Bitcoin dan harga emas selama periode 2020–2026?
3. Seberapa kuat hubungan antara harga Bitcoin dan harga minyak selama periode 2020–2026?
4. Bagaimana pola perubahan harga emas dan minyak ketika harga Bitcoin mengalami kenaikan atau penurunan selama periode 2020–2026?

## Dataset

Penelitian menggunakan tiga jenis data:

* **Bitcoin**: harga Bitcoin berdasarkan tanggal.
* **Emas**: harga emas berdasarkan tanggal.
* **Minyak**: harga minyak berdasarkan tanggal.

Data disimpan dalam database MySQL dengan tiga tabel:

```text
bitcoin_prices
gold_prices
oil_prices
```

## Metode

Tahapan penelitian meliputi:

1. Pengumpulan data historis.
2. Penyimpanan data menggunakan MySQL.
3. Data cleaning dan pengecekan missing value.
4. Penggabungan data berdasarkan tanggal.
5. Analisis statistik deskriptif.
6. Analisis tren harga.
7. Analisis korelasi.
8. Manual grouping berdasarkan kondisi Bitcoin naik atau turun.
9. Pemodelan menggunakan XGBoost.
10. Evaluasi hasil prediksi.

## Manual Grouping

Manual grouping digunakan untuk melihat pola perubahan emas dan minyak berdasarkan kondisi Bitcoin.

Perubahan harga harian Bitcoin dikelompokkan menjadi:

* **Naik**: perubahan harga > 0%
* **Turun**: perubahan harga < 0%
* **Tetap**: perubahan harga = 0%

Kemudian dihitung rata-rata perubahan harga Bitcoin, emas, dan minyak pada setiap kelompok.

## Hasil Analisis

### Korelasi Bitcoin dan Emas

Nilai korelasi:

```text
0.6981
```

Hasil tersebut menunjukkan hubungan **positif yang kuat**, sehingga Bitcoin dan emas cenderung bergerak searah selama periode penelitian.

### Korelasi Bitcoin dan Minyak

Nilai korelasi:

```text
-0.1934
```

Hasil tersebut menunjukkan hubungan **negatif yang sangat lemah**, sehingga perubahan harga minyak tidak memiliki hubungan yang kuat dengan perubahan harga Bitcoin.

### Hasil Manual Grouping

| Kondisi Bitcoin | Bitcoin |   Emas | Minyak |
| --------------- | ------: | -----: | -----: |
| Naik            |  +2,19% | +0,05% | +0,06% |
| Turun           |  -1,94% | +0,03% | +0,07% |

Hasil tersebut menunjukkan bahwa emas dan minyak tidak selalu mengikuti arah perubahan harian Bitcoin.

## Teknologi yang Digunakan

* Python
* Jupyter Notebook / Google Colab
* Pandas
* NumPy
* Matplotlib
* Seaborn
* XGBoost
* MySQL
* Apache Airflow

## Struktur Repository

```text
├── README.md
├── analisis_bitcoin.ipynb
└── data/
```

## Kesimpulan

Bitcoin memiliki pergerakan harga yang lebih fluktuatif dibandingkan emas dan minyak. Hubungan Bitcoin dengan emas menunjukkan korelasi positif yang kuat, sedangkan hubungan Bitcoin dengan minyak sangat lemah dan negatif. Berdasarkan manual grouping, perubahan harga emas dan minyak juga tidak selalu mengikuti arah pergerakan Bitcoin.

XGBoost digunakan sebagai metode machine learning untuk menguji kemampuan data historis Bitcoin, emas, dan minyak dalam melakukan prediksi
